"""DETR query alignment, feature-cache versioning and counterfactual example generation.
Stubbed DETR / BERT only (no downloads)."""

import random
import types

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

from configs.coco_labels import label_to_id
from configs.config import FEATURE_CACHE_VERSION
from datasets.counterfactual import generate_for_image, match_detections, mentioned_labels
from models.vision_encoder import select_detections


# ---------------------------------------------------------------- 0. DETR query alignment
def _fake_detr_outputs(seed=0, confident=(3, 17, 42, 88)):
    torch.manual_seed(seed)
    logits = torch.randn(100, 92)
    for q in confident:
        logits[q, q % 90 + 1] = 12.0
    boxes = torch.rand(100, 4) * 0.4 + 0.1
    hidden = torch.randn(100, 256)
    return logits, boxes, hidden


def test_select_detections_uses_the_query_that_produced_each_detection():
    logits, boxes, hidden = _fake_detr_outputs()
    picked = select_detections(logits, boxes, hidden, (640, 480), threshold=0.9, max_objects=12)
    assert sorted(picked["query_index"].tolist()) == [3, 17, 42, 88]
    # the embedding of every detection is its own query's state ...
    assert torch.equal(picked["embeddings"], hidden[picked["query_index"]])
    # ... not the state at its position in the filtered list (the old bug)
    assert not torch.equal(picked["embeddings"], hidden[: len(picked["query_index"])])
    # class ids come from the same query
    assert picked["label_ids"].tolist() == [q % 90 + 1 for q in picked["query_index"].tolist()]
    # sorted by confidence, at most max_objects
    assert torch.all(picked["scores"][:-1] >= picked["scores"][1:])
    assert len(select_detections(logits, boxes, hidden, (640, 480), max_objects=2)["scores"]) == 2


def test_select_detections_matches_official_post_processing():
    transformers = pytest.importorskip("transformers")
    processor = transformers.DetrImageProcessor()          # default config, no download
    logits, boxes, hidden = _fake_detr_outputs(seed=1)
    outputs = types.SimpleNamespace(logits=logits.unsqueeze(0), pred_boxes=boxes.unsqueeze(0))
    reference = processor.post_process_object_detection(outputs, threshold=0.9,
                                                        target_sizes=torch.tensor([[480, 640]]))[0]
    picked = select_detections(logits, boxes, hidden, (640, 480))
    order = torch.argsort(reference["scores"], descending=True)
    assert torch.allclose(reference["scores"][order], picked["scores"])
    assert torch.equal(reference["labels"][order], picked["label_ids"])
    assert torch.allclose(reference["boxes"][order], picked["boxes"], atol=1e-3)


def test_feature_cache_rebuilds_old_versions(tmp_path):
    import train

    calls = []

    class Vision:
        def __call__(self, path):
            calls.append(path)
            return {"embeddings": torch.ones(1, 2, 256), "labels": [["person", "dog"]],
                    "label_ids": [torch.tensor([1, 18])],
                    "boxes_norm": [torch.tensor([[0.1, 0.1, 0.5, 0.5], [0.4, 0.4, 0.9, 0.9]])]}

    model = types.SimpleNamespace(vision=Vision(), text=None)
    cache = train.FeatureCache(model, tmp_path)
    image = str(tmp_path / "img.jpg")
    old_file = cache.image_dir / f"{train._key(__import__('os').path.abspath(image))}.pt"
    torch.save({"embeddings": torch.zeros(2, 256), "labels": ["person", "dog"],
                "boxes": torch.zeros(2, 4)}, old_file)                     # version-1 file
    item = cache.image(image)
    assert calls == [image]                                                # rebuilt
    assert item["version"] == FEATURE_CACHE_VERSION
    assert item["label_ids"].tolist() == [1, 18] and torch.all(item["embeddings"] == 1)
    fresh = train.FeatureCache(model, tmp_path)
    fresh.image(image)
    assert calls == [image]                                                # new file reused


# ---------------------------------------------------------------- 1. counterfactual examples
LABELS = ["person", "bicycle", "car", "car", "dog", "bus"]
BOXES = torch.tensor([
    [0.40, 0.30, 0.52, 0.80],   # person
    [0.38, 0.55, 0.56, 0.90],   # bicycle (touches the person)
    [0.00, 0.50, 0.15, 0.62],   # car
    [0.70, 0.50, 0.85, 0.62],   # car
    [0.20, 0.80, 0.30, 0.95],   # dog
    [0.88, 0.05, 0.99, 0.30],   # bus
])
GROUND_TRUTH = [{"label": l, "box": b.tolist(), "iscrowd": 0} for l, b in zip(LABELS, BOXES)]
SUPER = {"person": "person", "bicycle": "vehicle", "motorcycle": "vehicle", "car": "vehicle",
         "bus": "vehicle", "truck": "vehicle", "dog": "animal", "cat": "animal"}
VOCAB = {"yes", "no", "0", "1", "2", "3", "bicycle", "motorcycle", "brown", "red", "cat"}
QUESTIONS = [
    {"question": "What is the man riding?", "answer": "bicycle", "question_id": 1},
    {"question": "What color is the dog?", "answer": "brown", "question_id": 2},
]


def _steer(embedding, label):
    return embedding + label_to_id(label), True


def _generate(ground_truth=GROUND_TRUTH, seed=0, caps=None):
    detections = {"labels": LABELS, "boxes": BOXES, "embeddings": torch.randn(len(LABELS), 256)}
    caps = caps or {"existence": 10, "counting": 10, "replacement": 10, "irrelevant": 10}
    return generate_for_image("img.jpg", detections, ground_truth, QUESTIONS, VOCAB, _steer, SUPER,
                              random.Random(seed), caps), detections


def test_match_detections_class_and_iou():
    noisy = BOXES.clone()
    noisy[2] += 0.3                                                         # moved far away
    matches = match_detections(LABELS, noisy, GROUND_TRUTH)
    assert matches[0] == 0 and matches[1] == 1 and matches[2] is None
    wrong_class = [dict(g, label="cat") if g["label"] == "dog" else g for g in GROUND_TRUTH]
    assert match_detections(LABELS, BOXES, wrong_class)[4] is None


def test_mentioned_labels():
    assert mentioned_labels("What is the man riding?") == {"person"}
    assert mentioned_labels("How many cars are near the bus?") == {"car", "bus"}
    assert mentioned_labels("Is the traffic light red?") == {"traffic light"}


def test_existence_examples():
    examples, _ = _generate()
    existence = [e for e in examples if e["type"] == "existence"]
    targets = [e for e in existence if e["subtype"] == "remove_target"]
    assert targets
    for e in targets:
        label = e["mentioned"][0]
        assert e["question"] in (f"Is there a {label}?", f"Is there an {label}?")
        assert e["factual_answer"] == "yes" and e["cf_answer"] == "no" and e["kind"] == "remove"
        assert sorted(e["nodes"]) == [i for i, l in enumerate(LABELS) if l == label]      # all instances
    others = [e for e in existence if e["subtype"] == "remove_other"]
    assert others and all(e["cf_answer"] == "yes" for e in others)
    for e in others:
        asked = e["mentioned"][0]
        assert all(LABELS[n] != asked for n in e["nodes"])


def test_counting_examples():
    examples, _ = _generate()
    cars = [e for e in examples if e["type"] == "counting" and e["question"] == "How many cars are there?"]
    assert len(cars) == 1
    assert cars[0]["factual_answer"] == "2" and cars[0]["cf_answer"] == "1"
    assert len(cars[0]["nodes"]) == 1 and LABELS[cars[0]["nodes"][0]] == "car"
    # a ground-truth car that DETR missed makes the count unreliable -> no counting example
    missed = GROUND_TRUTH + [{"label": "car", "box": [0.5, 0.0, 0.6, 0.1], "iscrowd": 0}]
    examples, _ = _generate(missed)
    assert not [e for e in examples if e["type"] == "counting" and "cars" in e["question"]]


def test_replacement_examples():
    examples, detections = _generate()
    replacement = [e for e in examples if e["type"] == "replacement"]
    assert len(replacement) == 1
    e = replacement[0]
    assert e["question"] == "What is the man riding?" and e["factual_answer"] == "bicycle"
    assert e["kind"] == "replace" and e["nodes"] == [1] and e["answer_node"] == 1
    assert e["cf_answer"] == e["new_label"] == "motorcycle"                # same super-category, in vocab
    assert e["new_label_id"] == label_to_id("motorcycle")
    assert torch.allclose(e["x_prime"][0], detections["embeddings"][1] + label_to_id("motorcycle"))


def test_irrelevant_examples_are_unrelated_and_keep_the_answer():
    examples, _ = _generate()
    irrelevant = [e for e in examples if e["type"] == "irrelevant"]
    assert irrelevant
    touching = {(0, 1), (1, 0)}
    for e in irrelevant:
        assert e["cf_answer"] == e["factual_answer"]
        node = e["nodes"][0]
        assert LABELS[node] not in e["mentioned"]
        mentioned_nodes = [i for i, l in enumerate(LABELS) if l in e["mentioned"]]
        assert not any((node, m) in touching for m in mentioned_nodes)
        if e["kind"] == "replace":
            assert e["new_label"] not in e["mentioned"] and e["x_prime"].shape == (1, 256)
