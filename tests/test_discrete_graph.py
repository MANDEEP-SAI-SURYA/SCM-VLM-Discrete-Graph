"""Discrete Causal Graph Sampling layer (--graph-mode discrete).  Stubbed DETR / BERT only."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import models.scm_vlm as scm_vlm_module
from configs.coco_labels import label_to_id
from losses.losses import acyclicity_loss
from models.scm_graph import SCMGraph

LABELS = ["person", "bicycle", "umbrella", "car", "dog", "bus"]
BOXES = torch.tensor([
    [0.40, 0.30, 0.52, 0.80], [0.38, 0.55, 0.56, 0.90], [0.39, 0.20, 0.54, 0.35],
    [0.00, 0.50, 0.20, 0.65], [0.70, 0.80, 0.80, 0.95], [0.85, 0.05, 0.99, 0.30],
])
CLASS_IDS = torch.tensor([label_to_id(label) for label in LABELS])


class StubVision(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()


class StubText(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

    def forward(self, questions):
        if isinstance(questions, str):
            questions = [questions]
        return torch.randn(len(questions), 768, generator=torch.Generator().manual_seed(7))


@pytest.fixture()
def make_model(monkeypatch):
    monkeypatch.setattr(scm_vlm_module, "VisionEncoder", StubVision)
    monkeypatch.setattr(scm_vlm_module, "TextEncoder", StubText)

    def make(**kwargs):
        torch.manual_seed(0)
        return scm_vlm_module.SCMVLM(num_classes=10, **kwargs)

    return make


def _forward(model, raw=None):
    raw = torch.randn(1, len(LABELS), 256, generator=torch.Generator().manual_seed(1)) if raw is None else raw
    q = torch.randn(1, 768, generator=torch.Generator().manual_seed(2))
    return model.forward_features(raw, q, BOXES.unsqueeze(0), CLASS_IDS.unsqueeze(0))


def test_discrete_graph_is_deterministic_and_sparse_at_test_time(make_model):
    model = make_model(graph_mode="discrete").eval()
    a, b = _forward(model), _forward(model)
    assert torch.equal(a["adjacency"], b["adjacency"])
    A = a["adjacency"][0]
    assert torch.all(torch.diagonal(A) == 0)
    assert torch.all((A > 0).sum(-1) <= model.graph.top_k)                  # at most k parents
    rows = A.sum(-1)
    assert torch.allclose(rows + a["null_weight"][0], torch.ones_like(rows), atol=1e-5)
    assert torch.all(rows <= 1 + 1e-6)


def test_parent_sets_are_sampled_while_training(make_model):
    model = make_model(graph_mode="discrete").train()
    model.graph.temperature = 1.0
    torch.manual_seed(0)
    patterns = {tuple((_forward(model)["adjacency"][0] > 0).flatten().tolist()) for _ in range(10)}
    assert len(patterns) > 1                                                 # Gumbel top-k explores


def test_straight_through_gradient_reaches_unselected_edges():
    torch.manual_seed(0)
    graph = SCMGraph(16, 8, top_k=2, mode="discrete").train()
    scores = torch.randn(1, 5, 5, requires_grad=True)
    masked = scores.masked_fill(torch.eye(5, dtype=torch.bool).unsqueeze(0), -1e9)
    sampled = graph._discrete_parents(masked, torch.randn(1, 5, 16), torch.randn(1, 16))
    adjacency = sampled["adjacency"]
    (adjacency * torch.randn_like(adjacency)).sum().backward()
    # edges allowed by the causal order but not chosen still receive the relaxed gradient
    unselected = (adjacency.detach() == 0) & sampled["allowed"]
    assert unselected.any()
    assert scores.grad[unselected].abs().sum() > 0
    assert graph.order_mlp[0].weight.grad.abs().sum() > 0                 # the causal order learns too


def test_no_parent_option_allows_a_true_dag(make_model):
    """The soft graph gives every object 3 parents, so it always contains a cycle (h(A) > 0).
    With the 'no parent' option the discrete graph can make objects root causes and become a DAG."""
    soft = make_model(graph_mode="soft").eval()
    assert float(acyclicity_loss(_forward(soft)["adjacency"][0].detach())) > 1e-3
    discrete = make_model(graph_mode="discrete").eval()
    with torch.no_grad():
        discrete.graph.null_mlp[-1].bias.fill_(50.0)          # every object prefers "no parent"
        discrete.graph.top_k = 1
    out = _forward(discrete)
    assert torch.all(out["adjacency"] == 0)
    assert float(acyclicity_loss(out["adjacency"][0])) == pytest.approx(0.0, abs=1e-6)


def test_temperature_schedule():
    from train import gumbel_temperature

    taus = [gumbel_temperature(e, 10, 1.0, 0.2) for e in range(10)]
    assert taus[0] == pytest.approx(1.0) and taus[-1] == pytest.approx(0.2)
    assert all(a > b for a, b in zip(taus, taus[1:]))


def test_remove_does_not_reinflate_the_remaining_parents(make_model):
    model = make_model(graph_mode="discrete").eval()
    out = _forward(model)
    A = out["adjacency"][0]
    parent, child = next((j, i) for i in range(len(LABELS)) for j in range(len(LABELS)) if A[i, j] > 0)
    cf = model.intervene(out, "remove", [parent])
    new_child = cf["keep"].index(child)
    assert float(cf["adjacency"][0, new_child].sum()) == pytest.approx(float(A[child].sum() - A[child, parent]),
                                                                       abs=1e-5)


@pytest.mark.parametrize("cf_mode", ["none", "invariance_old", "supervised"])
def test_one_training_step_discrete(make_model, cf_mode):
    import train

    model = make_model(graph_mode="discrete").train()
    settings = {"cf_mode": cf_mode, "lambda_cf": 1.0, "lambda_edge": 0.5, "irrelevant_graph_check": True}
    detections = {"embeddings": torch.randn(len(LABELS), 256), "boxes": BOXES, "labels": LABELS,
                  "label_ids": CLASS_IDS}
    q = torch.randn(768)
    loss, _, parts = train.factual_step(model, detections, q, torch.tensor(3), settings)
    assert "root_fraction" in parts
    if cf_mode == "supervised":
        example = {"type": "replacement", "kind": "replace", "nodes": [1], "new_label": "motorcycle",
                   "new_label_id": label_to_id("motorcycle"), "x_prime": torch.randn(1, 256),
                   "factual_answer": "bicycle", "cf_answer": "motorcycle", "mentioned": ["person", "bicycle"]}
        cf_loss, _, _ = train.counterfactual_step(model, detections, q, example,
                                                  {"bicycle": 2, "motorcycle": 3}, settings)
        loss = loss + cf_loss
    assert torch.isfinite(loss)
    loss.backward()
    assert model.graph.null_mlp[0].weight.grad.abs().sum() > 0
    assert model.graph.edge_mlp.net[0].weight.grad.abs().sum() > 0


def test_discrete_checkpoint_roundtrip(make_model, tmp_path):
    import train
    from models.scm_vlm import build_model_from_checkpoint

    model = make_model(graph_mode="discrete")
    model.graph.temperature = 0.2
    dataset = type("D", (), {"answer2idx": {str(i): i for i in range(10)},
                             "idx2answer": {i: str(i) for i in range(10)}})()
    path = tmp_path / "ckpt.pt"
    train.save_checkpoint(model, dataset, 1, 0.5, 30.0, str(path))
    checkpoint = torch.load(path, weights_only=False)
    rebuilt = build_model_from_checkpoint(checkpoint)
    assert rebuilt.architecture()["graph_mode"] == "discrete"
    assert rebuilt.graph.temperature == pytest.approx(0.2)
    result = rebuilt.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    assert not result.unexpected_keys
    assert not [k for k in result.missing_keys if not k.startswith(("vision.", "text."))]


def test_structure_metrics_and_whatif_on_discrete_graph(make_model):
    from models.do_intervention import DoInterventionReasoner, Intervention

    model = make_model(graph_mode="discrete").eval()
    reasoner = DoInterventionReasoner(model, {i: str(i) for i in range(10)}, steerer=None, mode="recompute")
    world = reasoner.observe_from_tensors(torch.randn(len(LABELS), 256), LABELS, "What is the man riding?",
                                          BOXES)
    result = reasoner.intervene(world, Intervention("remove", target="umbrella"))
    relations = reasoner.relationship_changes(result)
    assert "lost" in relations and result.counterfactual.adjacency.shape == (5, 5)


@pytest.mark.parametrize("training", [False, True])
def test_every_sampled_graph_is_a_dag(make_model, training):
    """The learned causal order makes acyclicity a guarantee, not a penalty."""
    model = make_model(graph_mode="discrete").train(training)
    torch.manual_seed(0)
    for seed in range(20):
        raw = torch.randn(1, len(LABELS), 256, generator=torch.Generator().manual_seed(seed))
        out = _forward(model, raw)
        A = out["adjacency"][0].detach()
        assert float(acyclicity_loss(A)) == pytest.approx(0.0, abs=1e-5)
        assert torch.all(A[~out["allowed_parents"][0]] == 0)          # only parents earlier in the order
        assert int((A.sum(-1) == 0).sum()) >= 1                        # at least one root cause


def test_parent_selection_targets():
    from train import parent_selection_loss

    boxes = torch.tensor([[0.0, 0.0, 0.2, 0.2], [0.1, 0.1, 0.3, 0.3], [0.8, 0.8, 0.9, 0.9]])  # 0-1 touch
    allowed = torch.tensor([[False, False, False], [True, False, False], [True, True, False]])
    good = torch.full((3, 4), -10.0)
    good[0, 3] = 10.0      # object 0: first in the order -> "no parent"
    good[1, 0] = 10.0      # object 1 touches object 0, which comes earlier -> parent 0
    good[2, 3] = 10.0      # object 2 touches nothing -> "no parent"
    bad = torch.full((3, 4), -10.0)
    bad[:, 3] = 10.0       # everything "no parent"
    assert parent_selection_loss(good, boxes, allowed) < parent_selection_loss(bad, boxes, allowed)


@pytest.mark.parametrize("diagnostic", [True, False])
def test_diagnostic_messages_carry_evidence_from_child_to_parent(make_model, diagnostic):
    """Same DAG, only the child's state changes: with diagnostic messages the parent's state
    changes too (evidence flows child -> parent); with causal messages only, it cannot."""
    model = make_model(graph_mode="discrete", diagnostic_messages=diagnostic).eval()
    out = _forward(model)
    A = out["adjacency"]
    parent, child = next((j, i) for i in range(len(LABELS)) for j in range(len(LABELS)) if A[0, i, j] > 0)
    objects = out["objects"].clone()
    # a random change (a constant shift would be removed by the blocks' LayerNorm)
    objects[0, child] += torch.randn(objects.shape[-1], generator=torch.Generator().manual_seed(3))
    boxes = BOXES.unsqueeze(0)
    before, _ = model.run_causal_layers(out["objects"], A, boxes, out["relation_probs"])
    after, _ = model.run_causal_layers(objects, A, boxes, out["relation_probs"])
    # every ancestor of the parent is a non-descendant of the child in a DAG, so with causal
    # messages only nothing reaches the parent
    parent_changed = not torch.allclose(after[0, parent], before[0, parent], atol=1e-6)
    assert parent_changed == diagnostic
    assert model.architecture()["diagnostic_messages"] == diagnostic


@pytest.mark.parametrize("kind", ["remove", "replace"])
def test_do_mode_recomputes_exactly_the_influenced_nodes(make_model, kind):
    """With diagnostic messages, do() changes the nodes connected to the intervened one in the
    mutilated graph; every other node must keep its exact state (no drift)."""
    from models.do_intervention import ClassSteerer, DoInterventionReasoner, Intervention
    from configs.coco_labels import LABEL2ID

    model = make_model(graph_mode="discrete").eval()
    steerer = ClassSteerer(nn.Linear(256, 92), dict(LABEL2ID))
    reasoner = DoInterventionReasoner(model, {i: str(i) for i in range(10)}, steerer, mode="do")
    world = reasoner.observe_from_tensors(torch.randn(len(LABELS), 256, generator=torch.Generator().manual_seed(4)),
                                          LABELS, "What is the man riding?", BOXES)
    intervention = Intervention(kind, target="umbrella", new_label="dog" if kind == "replace" else None)
    result = reasoner.intervene(world, intervention)
    assert result.non_descendant_max_drift < 1e-5
    out = _forward(model)
    cf = model.intervene(out, kind, [2], new_raw=torch.randn(1, 256), new_class_ids=[18]) if kind == "replace" \
        else model.intervene(out, kind, [2])
    keep = cf["keep"] or list(range(len(LABELS)))
    for factual_index, cf_index in zip(keep, range(len(keep))):
        if factual_index != 2 and factual_index not in cf["descendants"]:
            assert torch.allclose(cf["hidden"][0, cf_index], out["hidden"][0, factual_index], atol=1e-5)
