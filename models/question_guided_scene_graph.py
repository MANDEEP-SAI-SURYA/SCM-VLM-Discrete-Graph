"""Question-guided scene-graph construction and evidence retrieval."""

from __future__ import annotations

from dataclasses import dataclass
import re

import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class SceneNode:
    index: int
    label: str
    score: float
    box: tuple[float, float, float, float]
    relevance: float


@dataclass(frozen=True)
class SceneEdge:
    source: int
    relation: str
    target: int
    confidence: float


_ALIASES = {
    "person": {"person", "man", "woman", "boy", "girl", "people", "he", "she", "they"},
    "bicycle": {"bicycle", "bike", "cycling", "riding"},
    "motorcycle": {"motorcycle", "motorbike", "bike", "riding"},
    "umbrella": {"umbrella", "rain", "raining", "shade", "sun"},
    "traffic light": {"traffic", "light", "signal"},
    "car": {"car", "vehicle", "drive", "driving"},
    "bus": {"bus", "vehicle", "transport"},
}


class QuestionGuidedSceneGraph:
    """Build a spatial scene graph and retrieve question-relevant evidence.

    DETR provides nodes and boxes. BERT semantic similarity plus lexical
    matches provide a question-conditioned relevance score for each node.
    """

    def __init__(self, vision_encoder, text_encoder) -> None:
        self.vision_encoder = vision_encoder
        self.text_encoder = text_encoder

    @staticmethod
    def _tokens(text: str) -> set[str]:
        return set(re.findall(r"[a-z]+", text.lower()))

    @staticmethod
    def _iou(first: tuple[float, float, float, float], second: tuple[float, float, float, float]) -> float:
        left = max(first[0], second[0])
        top = max(first[1], second[1])
        right = min(first[2], second[2])
        bottom = min(first[3], second[3])
        intersection = max(0.0, right - left) * max(0.0, bottom - top)
        first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
        second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
        union = first_area + second_area - intersection
        return intersection / union if union else 0.0

    def _relevance_scores(self, labels: list[str], question: str) -> list[float]:
        question_tokens = self._tokens(question)
        with torch.inference_mode():
            question_embedding = self.text_encoder(question)
            label_embeddings = self.text_encoder([f"a {label}" for label in labels])
            semantic = F.cosine_similarity(label_embeddings, question_embedding.expand_as(label_embeddings))

        scores: list[float] = []
        for label, semantic_score in zip(labels, semantic.tolist()):
            label_tokens = self._tokens(label)
            aliases = _ALIASES.get(label.lower(), set())
            lexical_match = bool(question_tokens & (label_tokens | aliases))
            # Cosine similarity is useful for ranking but insufficient on its
            # own, so direct question/entity matches get a strong bonus.
            score = 0.70 * float(lexical_match) + 0.30 * ((semantic_score + 1.0) / 2.0)
            scores.append(round(score, 4))
        return scores

    @staticmethod
    def _pair_relation(first: SceneNode, second: SceneNode) -> tuple[str, str, float]:
        first_center = ((first.box[0] + first.box[2]) / 2, (first.box[1] + first.box[3]) / 2)
        second_center = ((second.box[0] + second.box[2]) / 2, (second.box[1] + second.box[3]) / 2)
        delta_x = second_center[0] - first_center[0]
        delta_y = second_center[1] - first_center[1]
        overlap = QuestionGuidedSceneGraph._iou(first.box, second.box)

        if overlap >= 0.10:
            return "overlaps", "overlaps", overlap
        if abs(delta_x) >= abs(delta_y):
            return ("left of", "right of", min(1.0, abs(delta_x) / 200.0)) if delta_x >= 0 else ("right of", "left of", min(1.0, abs(delta_x) / 200.0))
        return ("above", "below", min(1.0, abs(delta_y) / 200.0)) if delta_y >= 0 else ("below", "above", min(1.0, abs(delta_y) / 200.0))

    def build(self, image_path: str, question: str) -> tuple[list[SceneNode], list[SceneEdge]]:
        with torch.inference_mode():
            detection = self.vision_encoder(image_path)

        labels = detection["labels"][0]
        boxes = detection["boxes"][0].detach().cpu().tolist()
        detection_scores = detection["scores"][0].detach().cpu().tolist()
        relevance_scores = self._relevance_scores(labels, question)
        nodes = [
            SceneNode(index, label, score, tuple(box), relevance)
            for index, (label, box, score, relevance) in enumerate(
                zip(labels, boxes, detection_scores, relevance_scores)
            )
        ]

        edges: list[SceneEdge] = []
        for first_index, first in enumerate(nodes):
            for second in nodes[first_index + 1:]:
                forward, backward, confidence = self._pair_relation(first, second)
                edges.extend((
                    SceneEdge(first.index, forward, second.index, confidence),
                    SceneEdge(second.index, backward, first.index, confidence),
                ))
        return nodes, edges

    @staticmethod
    def retrieve(nodes: list[SceneNode], edges: list[SceneEdge], max_nodes: int = 4) -> tuple[list[SceneNode], list[SceneEdge]]:
        relevant_nodes = sorted(nodes, key=lambda node: node.relevance, reverse=True)[:max_nodes]
        selected_ids = {node.index for node in relevant_nodes}
        relevant_edges = [
            edge for edge in edges
            if edge.source in selected_ids and edge.target in selected_ids
        ]
        relevant_edges.sort(
            key=lambda edge: min(nodes[edge.source].relevance, nodes[edge.target].relevance),
            reverse=True,
        )
        return relevant_nodes, relevant_edges[:6]
