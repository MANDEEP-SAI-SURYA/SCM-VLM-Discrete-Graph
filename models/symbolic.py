"""Neuro-symbolic answer fusion (paper, Future Work: "neuro symbolic reasoning modules").

A rule-based reasoner (models/scene_answer.answer_from_scene) answers counting, existence,
colour and spatial-relation questions directly from the detected scene (labels + boxes).
Its answer becomes a one-hot evidence vector e over the answer vocabulary, and the neural
answer logits are fused with it through a learned, question-dependent gate:

    logits' = logits + g(q) * e,          g(q) = softplus(w^T q + b) >= 0

* questions the rules cannot parse, or whose rule answer is not in the vocabulary, give
  e = 0, so the neural answer is unchanged;
* the gate is learned from the question embedding, so the model learns for which question
  types to trust the symbolic answer (e.g. "how many" yes, "why" no);
* under an intervention the symbolic answer is recomputed on the EDITED scene (the removed
  object is gone, the replaced object has its new class), so the symbolic path is exactly
  consistent with do(): removing the only dog turns "Is there a dog?" into "no".

The evidence is computed outside the network (it needs the question text and the labels)
and passed to SCMVLM.forward_features / intervene as ``symbolic_prior`` [B, C].
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.coco_labels import COCO_LABELS
from models.scene_answer import SceneObject, answer_from_scene


def _resolver(known: list[str]):
    from models.do_intervention import resolve_label

    return lambda phrase: resolve_label(phrase, known)


class SymbolicAnswerer:
    """Maps (question, scene) -> evidence vector over the answer vocabulary."""

    def __init__(self, idx2answer) -> None:
        if isinstance(idx2answer, dict):
            items = {int(k): str(v) for k, v in idx2answer.items()}
            self.answers = [items[i] for i in range(len(items))]
        else:
            self.answers = [str(a) for a in idx2answer]
        self.answer2idx = {a: i for i, a in enumerate(self.answers)}
        self.resolve = _resolver(list(COCO_LABELS))

    def answer(self, question: str, labels: list[str], boxes=None):
        """The rule-based SceneAnswer or None."""
        objects = []
        for i, label in enumerate(labels):
            box = None
            if boxes is not None:
                box = tuple(float(v) for v in boxes[i])
            objects.append(SceneObject(i, str(label), box))
        try:
            return answer_from_scene(question, objects, self.resolve)
        except Exception:     # the rules must never break training
            return None

    def evidence(self, question: str, labels: list[str], boxes=None, device=None) -> torch.Tensor:
        """[1, C]: 1 at the symbolic answer, 0 elsewhere (all zeros = no symbolic evidence)."""
        vector = torch.zeros(1, len(self.answers), device=device)
        result = self.answer(question, labels, boxes)
        if result is not None and result.answer in self.answer2idx:
            vector[0, self.answer2idx[result.answer]] = 1.0
        return vector


class SymbolicFusion(nn.Module):
    """logits' = logits + softplus(w^T q + b) * evidence."""

    def __init__(self, question_dim: int) -> None:
        super().__init__()
        self.gate = nn.Linear(question_dim, 1)
        nn.init.zeros_(self.gate.weight)
        nn.init.constant_(self.gate.bias, 0.5)

    def gate_value(self, question_embedding: torch.Tensor) -> torch.Tensor:
        if question_embedding.dim() == 1:
            question_embedding = question_embedding.unsqueeze(0)
        return F.softplus(self.gate(question_embedding))          # [B, 1]

    def forward(self, logits, question_embedding, evidence):
        if evidence is None:
            return logits
        return logits + self.gate_value(question_embedding) * evidence.to(logits.dtype).to(logits.device)


def edited_scene(labels, boxes, kind: str, nodes, new_label: str | None = None):
    """Labels / boxes of the scene after the intervention (remove: the nodes are gone;
    replace: the nodes get ``new_label``, boxes unchanged)."""
    labels = list(labels)
    nodes = {int(n) for n in nodes}
    if boxes is not None and hasattr(boxes, "detach"):
        boxes = boxes.detach().cpu().reshape(-1, 4).tolist()
    if kind == "remove":
        keep = [i for i in range(len(labels)) if i not in nodes]
        return [labels[i] for i in keep], (None if boxes is None else [boxes[i] for i in keep])
    if kind == "replace":
        return [new_label if i in nodes else label for i, label in enumerate(labels)], boxes
    raise ValueError(f"unknown intervention kind: {kind}")
