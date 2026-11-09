"""Relation-aware messages, node identity, interventions, the no-graph ablation, one
training step per --cf-mode, evaluation statistics and checkpoint compatibility.
Stubbed DETR / BERT only (no downloads)."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import models.scm_vlm as scm_vlm_module
from configs.coco_labels import LABEL2ID, label_to_id
from models.causal_attention import MultiHeadCausalAttention
from models.scm_graph import pairwise_box_features

LABELS = ["person", "bicycle", "umbrella", "car", "dog"]
BOXES = torch.tensor([
    [0.40, 0.30, 0.52, 0.80],
    [0.38, 0.55, 0.56, 0.90],
    [0.39, 0.20, 0.54, 0.35],
    [0.00, 0.50, 0.20, 0.65],
    [0.70, 0.80, 0.80, 0.95],
])
CLASS_IDS = torch.tensor([label_to_id(label) for label in LABELS])


class StubVision(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()
        self.model = nn.Module()
        self.model.class_labels_classifier = nn.Linear(256, 92)
        self.model.config = type("C", (), {"label2id": dict(LABEL2ID)})()


class StubText(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

    def forward(self, questions):
        if isinstance(questions, str):
            questions = [questions]
        return torch.stack([torch.randn(768, generator=torch.Generator().manual_seed(len(q))) for q in questions])


@pytest.fixture()
def make_model(monkeypatch):
    monkeypatch.setattr(scm_vlm_module, "VisionEncoder", StubVision)
    monkeypatch.setattr(scm_vlm_module, "TextEncoder", StubText)

    def make(**kwargs):
        torch.manual_seed(0)
        return scm_vlm_module.SCMVLM(num_classes=12, **kwargs)

    return make


def _forward(model, raw=None, boxes=BOXES, class_ids=CLASS_IDS):
    torch.manual_seed(1)
    raw = torch.randn(1, len(LABELS), 256) if raw is None else raw
    return model.forward_features(raw, torch.randn(1, 768), boxes.unsqueeze(0), class_ids.unsqueeze(0))


# ---------------------------------------------------------------- 2. relation-aware messages
@pytest.mark.parametrize("relation_types", [1, 4])
def test_moving_a_parents_box_changes_the_childs_message(relation_types):
    torch.manual_seed(0)
    layer = MultiHeadCausalAttention(256, 8, dropout=0.0, relation_messages=True,
                                     relation_types=relation_types).eval()
    hidden = torch.randn(1, 3, 256)
    adjacency = torch.zeros(1, 3, 3)
    adjacency[0, 1, 0] = 1.0                       # 0 -> 1 (node 0 is the parent of node 1)
    probs = torch.softmax(torch.randn(1, 3, 3, relation_types), -1) if relation_types > 1 else None
    boxes = torch.tensor([[0.1, 0.1, 0.3, 0.3], [0.2, 0.2, 0.5, 0.5], [0.7, 0.7, 0.9, 0.9]])
    moved = boxes.clone()
    moved[0] = torch.tensor([0.6, 0.0, 0.8, 0.2])  # move only the parent
    out_a, _ = layer(hidden, adjacency, pairwise_box_features(boxes.unsqueeze(0)), probs)
    out_b, _ = layer(hidden, adjacency, pairwise_box_features(moved.unsqueeze(0)), probs)
    assert not torch.allclose(out_a[0, 1], out_b[0, 1])          # child's message changed
    assert torch.allclose(out_a[0, 2], out_b[0, 2])              # node 2 has no parents
    assert torch.allclose(out_a[0, 2], layer.out.bias)           # ... so it receives a zero message


def test_legacy_messages_ignore_geometry():
    torch.manual_seed(0)
    layer = MultiHeadCausalAttention(256, 8, dropout=0.0).eval()
    hidden, adjacency = torch.randn(1, 3, 256), torch.zeros(1, 3, 3)
    adjacency[0, 1, 0] = 1.0
    assert torch.allclose(layer(hidden, adjacency)[0], layer(hidden, adjacency, torch.randn(1, 3, 3, 9))[0])


# ---------------------------------------------------------------- 3. node identity
def test_class_and_box_change_the_node_input(make_model):
    model = make_model().eval()
    raw = torch.randn(1, 2, 256)
    base = model.encode_nodes(raw, torch.tensor([[1, 2]]), BOXES[:2].unsqueeze(0))
    other_class = model.encode_nodes(raw, torch.tensor([[1, 4]]), BOXES[:2].unsqueeze(0))
    moved = BOXES[:2].clone()
    moved[1] += 0.05
    other_box = model.encode_nodes(raw, torch.tensor([[1, 2]]), moved.unsqueeze(0))
    assert torch.allclose(base[0, 0], other_class[0, 0]) and not torch.allclose(base[0, 1], other_class[0, 1])
    assert not torch.allclose(base[0, 1], other_box[0, 1])


# ---------------------------------------------------------------- interventions: box / class consistency
def test_remove_keeps_boxes_and_classes_aligned(make_model):
    model = make_model().eval()
    out = _forward(model)
    cf = model.intervene(out, "remove", [1, 3])
    assert cf["keep"] == [0, 2, 4]
    assert cf["class_ids"][0].tolist() == CLASS_IDS[[0, 2, 4]].tolist()
    assert torch.equal(cf["boxes_norm"][0], BOXES[[0, 2, 4]])
    assert cf["adjacency"].shape == (1, 3, 3)
    rows = cf["adjacency"][0].sum(-1)
    assert torch.all((rows.abs() < 1e-6) | ((rows - 1).abs() < 1e-5))
    # the edited scene is answered exactly like a normal question on the remaining objects,
    # when every remaining object kept all its parents
    full = model.forward_features(out["raw_objects"][:, [0, 2, 4]], out["question_embedding"],
                                  BOXES[[0, 2, 4]].unsqueeze(0), CLASS_IDS[[0, 2, 4]].unsqueeze(0))
    recompute = model.intervene(out, "remove", [1, 3], mode="recompute")
    assert torch.allclose(recompute["logits"], full["logits"], atol=1e-5)


def test_replace_is_a_strict_do(make_model):
    model = make_model().eval()
    out = _forward(model)
    x_prime = torch.randn(1, 256)
    cf = model.intervene(out, "replace", [1], new_raw=x_prime, new_class_ids=[label_to_id("motorcycle")])
    assert cf["class_ids"][0, 1] == label_to_id("motorcycle") and cf["class_ids"][0, 0] == CLASS_IDS[0]
    assert torch.equal(cf["boxes_norm"], out["boxes_norm"])                    # box unchanged
    assert torch.all(cf["adjacency"][0, 1] == 0)                               # parents of k cut
    others = [i for i in range(len(LABELS)) if i != 1]
    assert torch.equal(cf["adjacency"][0, others], out["adjacency"][0, others])  # rest of the graph kept
    expected = model.encode_nodes(x_prime.unsqueeze(0), torch.tensor([[label_to_id("motorcycle")]]),
                                  BOXES[1:2].unsqueeze(0))[0, 0]
    assert torch.allclose(cf["objects"][0, 1], expected)
    # only descendants of k change
    non_desc = [i for i in range(len(LABELS)) if i != 1 and i not in cf["descendants"]]
    assert torch.allclose(cf["hidden"][0, non_desc], out["hidden"][0, non_desc], atol=1e-5)


# ---------------------------------------------------------------- 5. no-graph ablation
def test_no_graph_model_runs_and_nodes_are_independent(make_model):
    model = make_model(graph_mode="none").eval()
    assert model.graph is None
    out = _forward(model)
    assert torch.all(out["adjacency"] == 0)
    raw = out["raw_objects"].clone()
    raw[0, 3] += 1.0                                         # change another object
    other = _forward(model, raw=raw)
    assert torch.allclose(out["hidden"][0, 0], other["hidden"][0, 0])
    removed = model.intervene(out, "remove", [2])
    replaced = model.intervene(out, "replace", [2], new_raw=torch.randn(1, 256), new_class_ids=[18])
    assert removed["logits"].shape == replaced["logits"].shape == out["logits"].shape
    assert replaced["descendants"] == []


# ---------------------------------------------------------------- 1 + 5. one training step per --cf-mode
@pytest.mark.parametrize("graph_mode", ["soft", "none"])
@pytest.mark.parametrize("cf_mode", ["none", "invariance_old", "supervised"])
def test_one_training_step(make_model, cf_mode, graph_mode):
    import train

    model = make_model(graph_mode=graph_mode)
    model.train()
    settings = {"cf_mode": cf_mode, "lambda_cf": 1.0, "lambda_edge": 0.5 if graph_mode == "soft" else 0.0,
                "irrelevant_graph_check": True}
    torch.manual_seed(3)
    detections = {"embeddings": torch.randn(len(LABELS), 256), "boxes": BOXES, "labels": LABELS,
                  "label_ids": CLASS_IDS}
    q = torch.randn(768)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=0.01)
    loss, logits, parts = train.factual_step(model, detections, q, torch.tensor(3), settings)
    if cf_mode == "supervised":
        answer2idx = {"yes": 0, "no": 1, "bicycle": 2, "motorcycle": 3}
        example = {"type": "replacement", "kind": "replace", "nodes": [1], "new_label": "motorcycle",
                   "new_label_id": label_to_id("motorcycle"), "x_prime": torch.randn(1, 256),
                   "factual_answer": "bicycle", "cf_answer": "motorcycle", "mentioned": ["person", "bicycle"]}
        cf_loss, correct, cf_parts = train.counterfactual_step(model, detections, q, example, answer2idx, settings)
        assert correct in (0, 1) and float(cf_parts["cf"]) > 0
        loss = loss + cf_loss
    assert torch.isfinite(loss)
    optimizer.zero_grad()
    loss.backward()
    if graph_mode == "soft":
        grad = model.graph.edge_mlp.net[0].weight.grad
        assert grad is not None and grad.abs().sum() > 0          # the graph receives gradient
    if cf_mode == "invariance_old" and graph_mode == "soft":
        assert float(parts["cf"]) >= 0
    optimizer.step()


def test_irrelevant_example_skipped_when_related_in_graph(make_model):
    import train

    model = make_model().eval()
    detections = {"embeddings": torch.randn(len(LABELS), 256), "boxes": BOXES, "labels": LABELS,
                  "label_ids": CLASS_IDS}
    out = train.forward_detections(model, detections, torch.randn(768))
    A = out["adjacency"][0]
    person = 0
    related = [j for j in range(1, len(LABELS)) if A[person, j] > 1e-6 or A[j, person] > 1e-6]
    unrelated = [j for j in range(1, len(LABELS)) if j not in related]
    assert related
    example = {"type": "irrelevant", "nodes": [related[0]], "mentioned": ["person"]}
    assert train.irrelevant_is_related(out, detections, example)
    if unrelated:
        assert not train.irrelevant_is_related(out, detections, dict(example, nodes=[unrelated[0]]))


# ---------------------------------------------------------------- evaluation statistics
def test_mcnemar_and_scramble():
    import random
    from utils.evaluation import is_relational, mcnemar_exact, scramble_graph

    result = mcnemar_exact([1, 1, 1, 1, 1, 1, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0])
    assert result["only_first_correct"] == 6 and result["only_second_correct"] == 0
    assert result["p_value"] == pytest.approx(2 / 2 ** 6)
    assert mcnemar_exact([1, 0], [1, 0])["p_value"] == 1.0
    A = torch.softmax(torch.randn(1, 5, 5), -1) * (1 - torch.eye(5))
    A = A / A.sum(-1, keepdim=True)
    S, _ = scramble_graph(A, None, random.Random(0))
    assert torch.all(torch.diagonal(S[0]) == 0)
    assert torch.allclose(S[0].sum(-1), torch.ones(5), atol=1e-5)
    assert not torch.allclose(S, A)
    assert is_relational("What is the man riding?") and not is_relational("What color is the car?")


# ---------------------------------------------------------------- checkpoint compatibility
def test_checkpoint_architecture_roundtrip(make_model, tmp_path):
    import train
    from models.scm_vlm import build_model_from_checkpoint

    model = make_model(relation_types=4, residual_last_block=False)
    dataset = type("D", (), {"answer2idx": {str(i): i for i in range(12)},
                             "idx2answer": {i: str(i) for i in range(12)}})()
    path = tmp_path / "ckpt.pt"
    train.save_checkpoint(model, dataset, 1, 0.5, 30.0, str(path))
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["format_version"] == 3 and checkpoint["architecture"]["relation_types"] == 4
    rebuilt = build_model_from_checkpoint(checkpoint)
    assert rebuilt.architecture() == model.architecture()
    result = rebuilt.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    assert not result.unexpected_keys
    assert not [k for k in result.missing_keys if not k.startswith(("vision.", "text."))]
    # a format-2 checkpoint (before this change) maps to the old architecture
    old = {"num_classes": 12, "question_aware": True, "graph_mode": "soft", "graph_top_k": 3,
           "graph_geometry": True}
    legacy = build_model_from_checkpoint(old)
    assert legacy.architecture()["node_identity"] is False
    assert legacy.architecture()["relation_messages"] is False
    assert legacy.project_objects[0].in_features == 256
