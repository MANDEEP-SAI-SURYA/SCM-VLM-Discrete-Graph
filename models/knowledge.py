"""Symbolic knowledge base of class-pair relations and its injection into the edge scorer
(paper, Future Work: "integrating external symbolic knowledge bases or neuro symbolic
reasoning modules").

Knowledge base K[a, b, r] for DETR / COCO class ids a (subject) and b (object):

    cooccur   tanh(PMI(a, b)) of the two classes appearing in the same image
    touches   P(boxes of a and b touch or overlap | a and b in the same image)
    overlaps  P(IoU(a, b) > 0.1 | ...)
    above     P(centre of a is above the centre of b | ...)
    below     P(centre of a is below the centre of b | ...)
    inside    P(a's box lies inside b's box | ...)
    near      P(gap between a and b < 10 % of the image | ...)
    external  confidence of any triple (a, relation, b) in an external KB file
              (ConceptNet, Visual Genome relationships, hand-written rules ...)

The statistical relations are estimated from the COCO instance annotations of the TRAINING
images (validation images are excluded) with Bayesian smoothing towards the global rate, so
rare pairs do not get extreme values.  Every entry can be printed as a symbolic triple
(subject, relation, object, confidence).

External TSV format (one triple per line, '#' comments allowed):

    person<TAB>rides<TAB>bicycle<TAB>0.9
"""

from __future__ import annotations

import math
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn as nn

from configs.coco_labels import DETR_ID2LABEL, NUM_CLASS_IDS, label_to_id

KNOWLEDGE_RELATIONS = ("cooccur", "touches", "overlaps", "above", "below", "inside", "near", "external")
NUM_KNOWLEDGE_RELATIONS = len(KNOWLEDGE_RELATIONS)
KNOWLEDGE_VERSION = 1


def _pair_relations(a, b) -> dict[str, bool]:
    """Spatial relations of box a relative to box b (x0, y0, x1, y1 in 0-1)."""
    gap_x = max(0.0, max(a[0], b[0]) - min(a[2], b[2]))
    gap_y = max(0.0, max(a[1], b[1]) - min(a[3], b[3]))
    inter_w = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    inter_h = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = inter_w * inter_h
    area_a = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-9, (b[2] - b[0]) * (b[3] - b[1]))
    iou = inter / (area_a + area_b - inter)
    ca_y, cb_y = (a[1] + a[3]) / 2, (b[1] + b[3]) / 2
    return {
        "touches": gap_x <= 0.02 and gap_y <= 0.02,
        "overlaps": iou > 0.1,
        "above": ca_y < cb_y - 0.02,
        "below": ca_y > cb_y + 0.02,
        "inside": inter >= 0.9 * area_a,
        "near": max(gap_x, gap_y) < 0.1,
    }


def build_knowledge_base(ground_truth: dict[int, list[dict]], exclude_image_ids: set[int] | None = None,
                         prior_strength: float = 5.0) -> torch.Tensor:
    """K [NUM_CLASS_IDS, NUM_CLASS_IDS, NUM_KNOWLEDGE_RELATIONS] from COCO ground truth
    ({image_id: [{"label", "box" (0-1), "iscrowd"}]}, see datasets.counterfactual.load_coco_instances)."""
    exclude_image_ids = exclude_image_ids or set()
    spatial = KNOWLEDGE_RELATIONS[1:-1]
    pair_counts = defaultdict(float)
    rel_counts = defaultdict(float)
    class_images = defaultdict(float)
    pair_images = defaultdict(float)
    global_rel = defaultdict(float)
    global_pairs = 0.0
    images = 0
    for image_id, objects in ground_truth.items():
        if image_id in exclude_image_ids:
            continue
        objects = [o for o in objects if not o.get("iscrowd")]
        if not objects:
            continue
        images += 1
        ids = [label_to_id(o["label"]) for o in objects]
        for c in set(ids):
            class_images[c] += 1
        for a in set(ids):
            for b in set(ids):
                if a != b:
                    pair_images[(a, b)] += 1
        for i, oi in enumerate(objects):
            for j, oj in enumerate(objects):
                if i == j:
                    continue
                key = (ids[i], ids[j])
                pair_counts[key] += 1
                global_pairs += 1
                for name, value in _pair_relations(oi["box"], oj["box"]).items():
                    if value:
                        rel_counts[key + (name,)] += 1
                        global_rel[name] += 1
    knowledge = torch.zeros(NUM_CLASS_IDS, NUM_CLASS_IDS, NUM_KNOWLEDGE_RELATIONS)
    if images == 0:
        return knowledge
    for (a, b), n in pair_counts.items():
        for r, name in enumerate(spatial, start=1):
            prior = global_rel[name] / max(global_pairs, 1.0)
            knowledge[a, b, r] = (rel_counts[(a, b, name)] + prior_strength * prior) / (n + prior_strength)
    for (a, b), n in pair_images.items():
        p_ab = n / images
        p_a, p_b = class_images[a] / images, class_images[b] / images
        knowledge[a, b, 0] = math.tanh(math.log(max(p_ab, 1e-9) / max(p_a * p_b, 1e-12)))
    return knowledge


def load_external_triples(path: str | Path, knowledge: torch.Tensor) -> tuple[torch.Tensor, int]:
    """Add the triples of a TSV file (subject, relation, object, confidence) to the 'external'
    channel (the maximum confidence over relations).  Unknown class names are skipped."""
    knowledge = knowledge.clone()
    channel = KNOWLEDGE_RELATIONS.index("external")
    added = 0
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t")
        if len(parts) < 3:
            continue
        subject, obj = label_to_id(parts[0].strip()), label_to_id(parts[2].strip())
        if subject >= 91 or obj >= 91:
            continue
        confidence = float(parts[3]) if len(parts) > 3 and parts[3].strip() else 1.0
        knowledge[subject, obj, channel] = max(float(knowledge[subject, obj, channel]), confidence)
        added += 1
    return knowledge, added


def knowledge_triples(knowledge: torch.Tensor, min_confidence: float = 0.5, top: int | None = None):
    """Readable symbolic triples (subject, relation, object, confidence)."""
    triples = []
    nonzero = torch.nonzero(knowledge >= min_confidence, as_tuple=False)
    for a, b, r in nonzero.tolist():
        if r == 0:          # co-occurrence is a PMI score, not a probability
            continue
        triples.append((DETR_ID2LABEL.get(a, "?"), KNOWLEDGE_RELATIONS[r], DETR_ID2LABEL.get(b, "?"),
                        float(knowledge[a, b, r])))
    triples.sort(key=lambda t: -t[3])
    return triples[:top] if top else triples


class KnowledgePrior(nn.Module):
    """Edge-score prior from the knowledge base: for a candidate edge j -> i (A[i, j]),
    prior_ij = gate * MLP(K[class_i, class_j]).  The knowledge tensor is a buffer (saved with
    the checkpoint), the MLP and the gate are learned."""

    def __init__(self, hidden: int = 32) -> None:
        super().__init__()
        self.register_buffer("knowledge", torch.zeros(NUM_CLASS_IDS, NUM_CLASS_IDS, NUM_KNOWLEDGE_RELATIONS))
        # input: K[class_i, class_j] and K[class_j, class_i] (both directions of the pair)
        self.mlp = nn.Sequential(nn.Linear(2 * NUM_KNOWLEDGE_RELATIONS, hidden), nn.ReLU(), nn.Linear(hidden, 1))
        self.gate = nn.Parameter(torch.tensor(0.5))

    def set_knowledge(self, knowledge: torch.Tensor) -> None:
        self.knowledge.copy_(knowledge.to(self.knowledge.device))

    def forward(self, class_ids: torch.Tensor) -> torch.Tensor:
        """class_ids [B, N] -> prior logits [B, N, N] (row = receiver i, column = sender j)."""
        class_ids = class_ids.long().clamp(0, NUM_CLASS_IDS - 1)
        forward = self.knowledge[class_ids[:, :, None], class_ids[:, None, :]]       # [B, N, N, R]
        features = torch.cat([forward, forward.transpose(1, 2)], dim=-1)
        return self.gate * self.mlp(features).squeeze(-1)
