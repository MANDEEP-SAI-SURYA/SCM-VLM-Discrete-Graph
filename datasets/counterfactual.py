"""Counterfactual training / evaluation examples with KNOWN answers, built from VQA v2 and
the COCO instance annotations (no extra dataset).

Every example is (question, intervention, counterfactual answer):

  existence    "Is there a {class}?"   remove every node of that class      -> "no"
               (subtype "keep")        remove every node of ANOTHER class   -> "yes"
  counting     "How many {class}s are there?"  (1-5 instances, every detection and every
               ground-truth object of the class matched)  remove one         -> count - 1
  replacement  a VQA question whose answer is a COCO class name matching exactly one detected
               node, e.g. "What is the man riding?" -> "bicycle":
               do(X_k = x'), x' = the node's DETR embedding steered (ClassSteerer) to a class of
               the same COCO super-category whose name is in the answer vocabulary
                                                                      -> the new class name
  irrelevant   a VQA question; remove or replace an object that is not mentioned in the
               question/answer and does not touch a mentioned object  -> answer unchanged
               (train.py additionally skips the example when that node is a parent or child
               of a mentioned object in the CURRENT learned graph)

"remove" deletes the node and its edges; "replace" uses the strict do() path (incoming edges
cut, rest of the graph kept).  See SCMVLM.intervene.

DETR detections are matched to COCO ground truth (same class, IoU >= 0.5, greedy by IoU), so
existence and counting targets are only generated for objects that are really there.
"""

from __future__ import annotations

import hashlib
import json
import random
import re
from collections import defaultdict
from pathlib import Path
from typing import Callable

import torch

from configs.coco_labels import COCO_LABELS, LABEL2ID, label_to_id, plural, with_article
from models.scm_graph import contact_targets

GENERATOR_VERSION = 1
CF_TYPES = ("existence", "counting", "replacement", "irrelevant")
DEFAULT_CAPS = {"existence": 1, "counting": 1, "replacement": 2, "irrelevant": 2}

# words that refer to a COCO class in questions (plus the class names themselves)
_SYNONYMS = {
    "man": "person", "men": "person", "woman": "person", "women": "person", "people": "person",
    "boy": "person", "girl": "person", "child": "person", "children": "person", "kid": "person",
    "kids": "person", "he": "person", "she": "person", "player": "person", "lady": "person",
    "guy": "person", "rider": "person", "skier": "person", "surfer": "person",
    "bike": "bicycle", "bikes": "bicycle", "motorbike": "motorcycle", "plane": "airplane",
    "jet": "airplane", "sofa": "couch", "table": "dining table", "phone": "cell phone",
    "television": "tv", "computer": "laptop", "doughnut": "donut", "puppy": "dog",
    "kitten": "cat", "racket": "tennis racket", "glove": "baseball glove", "bat": "baseball bat",
    "ball": "sports ball", "plant": "potted plant", "fridge": "refrigerator", "light": "traffic light",
    "hydrant": "fire hydrant", "glass": "wine glass", "bag": "handbag", "purse": "handbag",
}


# =====================================================================
# COCO ground truth
# =====================================================================

_INSTANCES_MEMO: dict[str, dict] = {}


def load_coco_instances(path: str | Path, image_ids: set[int] | None = None):
    """Ground truth per image: {image_id: [{"label", "box" (x0, y0, x1, y1 in 0-1), "iscrowd"}]}
    and {label: supercategory}.  The JSON (~330 MB for train2014) is parsed once per process."""
    key = str(Path(path).resolve())
    if key not in _INSTANCES_MEMO:
        with Path(path).open("r", encoding="utf-8") as file:
            _INSTANCES_MEMO[key] = json.load(file)
    data = _INSTANCES_MEMO[key]
    categories = {c["id"]: c for c in data["categories"]}
    sizes = {img["id"]: (img["width"], img["height"]) for img in data["images"]
             if image_ids is None or img["id"] in image_ids}
    ground_truth: dict[int, list[dict]] = defaultdict(list)
    for ann in data["annotations"]:
        image_id = ann["image_id"]
        if image_id not in sizes:
            continue
        width, height = sizes[image_id]
        x, y, w, h = ann["bbox"]
        ground_truth[image_id].append({
            "label": categories[ann["category_id"]]["name"],
            "box": [x / width, y / height, (x + w) / width, (y + h) / height],
            "iscrowd": int(ann.get("iscrowd", 0)),
        })
    supercategory = {c["name"]: c["supercategory"] for c in data["categories"]}
    return dict(ground_truth), supercategory


def box_iou(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """IoU matrix between boxes a [N, 4] and b [M, 4] (x0, y0, x1, y1)."""
    if len(a) == 0 or len(b) == 0:
        return torch.zeros(len(a), len(b))
    left = torch.maximum(a[:, None, 0], b[None, :, 0])
    top = torch.maximum(a[:, None, 1], b[None, :, 1])
    right = torch.minimum(a[:, None, 2], b[None, :, 2])
    bottom = torch.minimum(a[:, None, 3], b[None, :, 3])
    inter = (right - left).clamp(min=0) * (bottom - top).clamp(min=0)
    area_a = ((a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1]))[:, None]
    area_b = ((b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1]))[None, :]
    return inter / (area_a + area_b - inter).clamp(min=1e-9)


def match_detections(det_labels: list[str], det_boxes: torch.Tensor, ground_truth: list[dict],
                     iou_threshold: float = 0.5) -> list[int | None]:
    """For every detection, the index of the matched ground-truth object (same class,
    IoU >= threshold, greedy by IoU, one-to-one) or None."""
    matches: list[int | None] = [None] * len(det_labels)
    if not ground_truth or len(det_labels) == 0:
        return matches
    gt_boxes = torch.tensor([g["box"] for g in ground_truth], dtype=torch.float)
    iou = box_iou(det_boxes.float(), gt_boxes)
    pairs = [(float(iou[d, g]), d, g) for d in range(len(det_labels)) for g in range(len(ground_truth))
             if det_labels[d] == ground_truth[g]["label"] and float(iou[d, g]) >= iou_threshold]
    used_d, used_g = set(), set()
    for _, d, g in sorted(pairs, reverse=True):
        if d not in used_d and g not in used_g:
            matches[d] = g
            used_d.add(d)
            used_g.add(g)
    return matches


# =====================================================================
# question analysis
# =====================================================================

def mentioned_labels(text: str) -> set[str]:
    """COCO classes referred to in a question or answer (names, simple plurals, synonyms)."""
    words = re.findall(r"[a-z]+", text.lower())
    found = set()
    for n in (2, 1):
        for i in range(len(words) - n + 1):
            phrase = " ".join(words[i:i + n])
            candidates = [phrase, phrase[:-1] if phrase.endswith("s") else phrase,
                          phrase[:-2] if phrase.endswith("es") else phrase]
            for candidate in candidates:
                if candidate in LABEL2ID:
                    found.add(candidate)
                elif candidate in _SYNONYMS:
                    found.add(_SYNONYMS[candidate])
    return found


# =====================================================================
# generation
# =====================================================================

def _example(kind_type, subtype, image_path, question, factual_answer, cf_answer, intervention,
             nodes, new_label=None, x_prime=None, mentioned=(), answer_node=None, question_id=None):
    return {
        "type": kind_type,
        "subtype": subtype,
        "image_path": str(image_path),
        "question": question,
        "question_id": question_id,
        "factual_answer": factual_answer,
        "cf_answer": cf_answer,
        "kind": intervention,                 # "remove" or "replace"
        "nodes": [int(n) for n in nodes],
        "new_label": new_label,
        "new_label_id": label_to_id(new_label) if new_label else None,
        "x_prime": x_prime,                   # [len(nodes), 256] steered DETR embedding (replace)
        "mentioned": sorted(mentioned),
        "answer_node": answer_node,
    }


def generate_for_image(
    image_path: str,
    detections: dict,
    ground_truth: list[dict],
    questions: list[dict],
    vocab: set[str],
    steer: Callable[[torch.Tensor, str], tuple[torch.Tensor, bool]] | None,
    supercategory: dict[str, str],
    rng: random.Random,
    caps: dict | None = None,
) -> list[dict]:
    """Counterfactual examples for one image.

    detections: {"labels": [str], "boxes": [N, 4] in 0-1, "embeddings": [N, 256]}
    questions:  VQA records of this image ({"question", "answer", "question_id", ...})
    steer(embedding, label) -> (x', converged): ClassSteerer.steer wrapper (replace examples)
    """
    caps = {**DEFAULT_CAPS, **(caps or {})}
    labels = list(detections["labels"])
    boxes = detections["boxes"].float()
    embeddings = detections["embeddings"]
    n = len(labels)
    if n == 0:
        return []
    matches = match_detections(labels, boxes, ground_truth)
    by_label = defaultdict(list)
    for i, label in enumerate(labels):
        by_label[label].append(i)
    touching = contact_targets(boxes) > 0
    examples: list[dict] = []

    # ---------------- existence ----------------
    if {"yes", "no"} <= vocab and caps["existence"] > 0:
        present = [c for c, idx in by_label.items() if any(matches[i] is not None for i in idx)]
        rng.shuffle(present)
        for target in present[:caps["existence"]]:
            question = f"Is there {with_article(target)}?"
            target_nodes = by_label[target]
            if len(target_nodes) < n:
                examples.append(_example("existence", "remove_target", image_path, question, "yes", "no",
                                         "remove", target_nodes, mentioned={target}))
            others = [c for c in by_label if c != target and len(by_label[c]) < n]
            if others:
                other = rng.choice(sorted(others))
                examples.append(_example("existence", "remove_other", image_path, question, "yes", "yes",
                                         "remove", by_label[other], mentioned={target}))

    # ---------------- counting ----------------
    if caps["counting"] > 0:
        countable = []
        for c, idx in by_label.items():
            gt_c = [g for g in ground_truth if g["label"] == c]
            if any(g["iscrowd"] for g in gt_c):
                continue
            k = len(idx)
            if not 1 <= k <= 5 or len(gt_c) != k or any(matches[i] is None for i in idx):
                continue
            if str(k) in vocab and str(k - 1) in vocab and k < n:
                countable.append(c)
        rng.shuffle(countable)
        for c in countable[:caps["counting"]]:
            k = len(by_label[c])
            removed = rng.choice(by_label[c])
            examples.append(_example("counting", "remove_one", image_path, f"How many {plural(c)} are there?",
                                     str(k), str(k - 1), "remove", [removed], mentioned={c}))

    # ---------------- replacement ----------------
    if steer is not None and caps["replacement"] > 0:
        made = 0
        for record in questions:
            if made >= caps["replacement"]:
                break
            answer = record["answer"]
            if answer not in LABEL2ID or len(by_label.get(answer, [])) != 1:
                continue
            node = by_label[answer][0]
            if matches[node] is None:
                continue
            group = supercategory.get(answer)
            options = sorted(c for c in COCO_LABELS
                             if c != answer and c in vocab and supercategory.get(c) == group)
            if not options:
                continue
            new_label = rng.choice(options)
            x_prime, converged = steer(embeddings[node], new_label)
            if not converged:
                continue
            mentioned = mentioned_labels(record["question"]) | {answer}
            examples.append(_example("replacement", "class_swap", image_path, record["question"], answer,
                                     new_label, "replace", [node], new_label,
                                     x_prime.detach().reshape(1, -1).cpu(), mentioned, node,
                                     record.get("question_id")))
            made += 1

    # ---------------- irrelevant (answer must not change) ----------------
    if caps["irrelevant"] > 0:
        made = 0
        shuffled = list(questions)
        rng.shuffle(shuffled)
        for record in shuffled:
            if made >= caps["irrelevant"]:
                break
            if record["answer"] not in vocab:
                continue
            mentioned = mentioned_labels(record["question"]) | mentioned_labels(record["answer"])
            mentioned_nodes = [i for i in range(n) if labels[i] in mentioned]
            if not mentioned_nodes:
                continue          # we cannot tell what is relevant to this question
            candidates = [j for j in range(n) if labels[j] not in mentioned
                          and not any(bool(touching[j, m]) for m in mentioned_nodes)]
            if not candidates or n < 2:
                continue
            node = rng.choice(candidates)
            if steer is not None and rng.random() < 0.5:
                options = sorted(c for c in COCO_LABELS if c not in mentioned and c != labels[node])
                new_label = rng.choice(options)
                x_prime, converged = steer(embeddings[node], new_label)
                if converged:
                    examples.append(_example("irrelevant", "replace", image_path, record["question"],
                                             record["answer"], record["answer"], "replace", [node], new_label,
                                             x_prime.detach().reshape(1, -1).cpu(), mentioned,
                                             question_id=record.get("question_id")))
                    made += 1
                    continue
            examples.append(_example("irrelevant", "remove", image_path, record["question"], record["answer"],
                                     record["answer"], "remove", [node], mentioned=mentioned,
                                     question_id=record.get("question_id")))
            made += 1

    return examples


def examples_cache_key(image_paths: list[str], vocab: list[str], caps: dict, seed: int,
                       feature_cache_version: int) -> str:
    digest = hashlib.md5()
    digest.update(f"v{GENERATOR_VERSION}|f{feature_cache_version}|s{seed}|{sorted(caps.items())}".encode())
    for path in sorted(image_paths):
        digest.update(path.encode())
    for answer in vocab:
        digest.update(answer.encode())
    return digest.hexdigest()[:16]


def build_counterfactual_examples(
    records: list[dict],
    detections_fn: Callable[[str], dict],
    instances_path: str | Path,
    vocab: list[str],
    steer: Callable[[torch.Tensor, str], tuple[torch.Tensor, bool]] | None,
    cache_dir: str | Path,
    split_name: str,
    seed: int = 42,
    caps: dict | None = None,
    feature_cache_version: int = 2,
    verbose: bool = True,
) -> list[dict]:
    """Generate (or load from cache) the counterfactual examples for the images of ``records``.

    records:       VQA records of one split (dicts with image_path, image_id, question, answer)
    detections_fn: image_path -> {"labels", "boxes", "embeddings"} (FeatureCache)
    """
    caps = {**DEFAULT_CAPS, **(caps or {})}
    if not records:
        return []
    by_image: dict[str, list[dict]] = defaultdict(list)
    for record in records:
        by_image[str(record["image_path"])].append(record)
    key = examples_cache_key(list(by_image), vocab, caps, seed, feature_cache_version)
    cache_path = Path(cache_dir) / f"counterfactual_{split_name}_{key}.pt"
    if cache_path.is_file():
        examples = torch.load(cache_path, map_location="cpu", weights_only=False)
        if verbose:
            print(f"Loaded {len(examples)} counterfactual examples ({split_name}) from {cache_path}")
        return examples

    image_ids = {int(r[0]["image_id"]) for r in by_image.values()}
    ground_truth, supercategory = load_coco_instances(instances_path, image_ids)
    vocab_set = set(vocab)
    rng = random.Random(seed)
    examples: list[dict] = []
    for count, (image_path, image_records) in enumerate(sorted(by_image.items()), start=1):
        try:
            detections = detections_fn(image_path)
        except Exception as error:  # unreadable image
            if verbose:
                print(f"CF SKIP {image_path}: {error}")
            continue
        examples.extend(generate_for_image(
            image_path, detections, ground_truth.get(int(image_records[0]["image_id"]), []),
            image_records, vocab_set, steer, supercategory, rng, caps,
        ))
        if verbose and count % 1000 == 0:
            print(f"[CF] {split_name}: {count}/{len(by_image)} images, {len(examples)} examples")
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(examples, cache_path)
    if verbose:
        counts = {t: sum(e["type"] == t for e in examples) for t in CF_TYPES}
        print(f"Built {len(examples)} counterfactual examples ({split_name}): {counts} -> {cache_path}")
    return examples
