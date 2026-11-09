"""Tests for do() what-if interventions. No DETR/BERT download needed:
the encoders are replaced with small stubs."""

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import models.scm_vlm as scm_vlm_module
from configs.coco_labels import label_to_id
from models.do_intervention import (
    ClassSteerer,
    DoInterventionReasoner,
    Intervention,
    descendants_of,
    parse_do_spec,
    parse_whatif,
    resolve_label,
)

LABELS = ["person", "bicycle", "umbrella", "car", "car", "traffic light"]
DETR_LABELS = ["person", "bicycle", "car", "motorcycle", "bus", "traffic light", "umbrella", "dog"]


class StubVision(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

    def forward(self, image_paths):
        raise RuntimeError("not used in tests")


class StubText(nn.Module):
    def __init__(self, freeze=True):
        super().__init__()

    def forward(self, questions):
        if isinstance(questions, str):
            questions = [questions]
        g = torch.Generator().manual_seed(len(questions[0]))
        return torch.randn(len(questions), 768, generator=g)


@pytest.fixture()
def reasoner(monkeypatch):
    monkeypatch.setattr(scm_vlm_module, "VisionEncoder", StubVision)
    monkeypatch.setattr(scm_vlm_module, "TextEncoder", StubText)
    torch.manual_seed(0)
    model = scm_vlm_module.SCMVLM(num_classes=10).eval()   # soft graph with box geometry
    classifier = nn.Linear(256, len(DETR_LABELS))
    steerer = ClassSteerer(classifier, {name: i for i, name in enumerate(DETR_LABELS)})
    # the tests above this file's "recompute" section check strict do() semantics
    return DoInterventionReasoner(model, {i: f"ans{i}" for i in range(10)}, steerer, mode="do")


# boxes (x0, y0, x1, y1) scaled to 0-1, one per object in LABELS
BOXES = torch.tensor([
    [0.40, 0.30, 0.52, 0.80],   # person
    [0.38, 0.55, 0.56, 0.90],   # bicycle (under the person)
    [0.39, 0.20, 0.54, 0.35],   # umbrella (above the person)
    [0.00, 0.50, 0.20, 0.65],   # car
    [0.65, 0.50, 0.82, 0.65],   # car
    [0.70, 0.05, 0.74, 0.20],   # traffic light
])


@pytest.fixture()
def world(reasoner):
    torch.manual_seed(1)
    return reasoner.observe_from_tensors(torch.randn(len(LABELS), 256), LABELS, "What is the person holding?",
                                         BOXES)


# ---------------------------------------------------------------- parsing
@pytest.mark.parametrize("text,kind,target,new,question", [
    ("What if there were no umbrella, what is the person holding?", "remove", "umbrella", None,
     "what is the person holding?"),
    ("What if the bicycle were a motorcycle? What is the man riding?", "replace", "the bicycle",
     "motorcycle", "what is the man riding?"),
    ("What if we removed the cars - is the road busy?", "remove", "the cars", None, "is the road busy?"),
    ("What if there was a dog?", "add", None, "dog", None),
    ("what if the umbrella is replaced with a bicycle, what is he holding", "replace", "the umbrella",
     "a bicycle", "what is he holding?"),
])
def test_parse_whatif(text, kind, target, new, question):
    intervention, q = parse_whatif(text)
    assert intervention.kind == kind
    assert intervention.target == target
    assert intervention.new_label == new
    assert q == question


def test_parse_do_spec():
    assert parse_do_spec("remove:car").kind == "remove"
    spec = parse_do_spec("replace:#2->dog")
    assert spec.nodes == [2] and spec.new_label == "dog"


def test_resolve_label():
    assert resolve_label("the cars", DETR_LABELS) == "car"
    assert resolve_label("a man", DETR_LABELS) == "person"
    assert resolve_label("bike", DETR_LABELS) == "bicycle"
    assert resolve_label("traffic lights", DETR_LABELS) == "traffic light"
    assert resolve_label("spaceship", DETR_LABELS) is None


# ---------------------------------------------------------------- graph semantics
def test_descendants_follow_message_direction():
    # A[i, j] > 0 means j -> i.  Chain 0 -> 1 -> 2, node 3 isolated.
    a = torch.zeros(4, 4)
    a[1, 0] = 1.0
    a[2, 1] = 1.0
    assert descendants_of(a, [0]) == [1, 2]
    assert descendants_of(a, [2]) == []


def test_replace_cuts_parents_and_keeps_non_descendants(reasoner, world):
    result = reasoner.intervene(world, Intervention("replace", target="umbrella", new_label="dog"))
    k = LABELS.index("umbrella")
    assert torch.all(result.counterfactual.adjacency[k] == 0)               # Eq. 12: incoming cut
    other_rows = [i for i in range(len(LABELS)) if i != k]
    assert torch.equal(result.counterfactual.adjacency[other_rows], world.adjacency[other_rows])
    assert result.counterfactual.labels[k] == "dog"
    assert result.steering[k]["converged"]
    for i in result.unaffected:                                             # Eq. 14
        assert torch.equal(result.counterfactual.hidden[i], world.hidden[i])
    assert result.non_descendant_max_drift < 1e-5
    for d in result.descendants:
        assert d not in result.unaffected


def test_remove_all_instances(reasoner, world):
    result = reasoner.intervene(world, Intervention("remove", target="cars"))
    assert result.intervened_nodes == [3, 4]
    assert "car" not in result.counterfactual.labels
    assert result.counterfactual.adjacency.shape == (4, 4)
    assert result.non_descendant_max_drift < 1e-5


def test_add_creates_exogenous_node(reasoner, world):
    result = reasoner.intervene(world, Intervention("add", new_label="dog"))
    n = len(LABELS)
    assert result.counterfactual.labels[-1] == "dog"
    assert torch.all(result.counterfactual.adjacency[n] == 0)               # no parents
    assert result.non_descendant_max_drift < 1e-5


def test_no_intervention_effect_without_descendants_is_consistent(reasoner, world):
    summary = reasoner.summarise(reasoner.intervene(world, Intervention("remove", nodes=[0])))
    assert 0.0 <= summary["total_variation"] <= 1.0
    assert summary["intervened_nodes"] == ["0:person"]


def test_missing_object_raises(reasoner, world):
    with pytest.raises(ValueError):
        reasoner.intervene(world, Intervention("remove", target="bus"))


# ---------------------------------------------------------------- attribute interventions
@pytest.mark.parametrize("text,target,value,question", [
    ("what if the traffic light turns green", "the traffic light", "green", None),
    ("What if the light turned red, should the cyclist stop?", "the light", "red", "should the cyclist stop?"),
    ("what if the car was blue", "the car", "blue", None),
])
def test_parse_attribute(text, target, value, question):
    intervention, q = parse_whatif(text)
    assert (intervention.kind, intervention.target, intervention.new_label, q) == ("attribute", target, value, question)


def test_attribute_default_question():
    intervention, _ = parse_whatif("what if the traffic light turns green")
    assert intervention.default_question() == "What color is the traffic light?"


def test_attribute_intervention_keeps_class_and_cuts_parents(reasoner, world):
    result = reasoner.intervene(world, Intervention("attribute", target="the light", new_label="green"))
    k = LABELS.index("traffic light")
    assert result.intervened_nodes == [k]
    assert result.counterfactual.labels[k] == "green traffic light"
    assert torch.all(result.counterfactual.adjacency[k] == 0)
    info = result.steering[k]
    assert info["similarity_after"] > info["similarity_before"]
    assert result.non_descendant_max_drift < 1e-5


def test_attribute_that_is_a_class_becomes_replace(reasoner, world):
    result = reasoner.intervene(world, Intervention("attribute", target="bicycle", new_label="motorcycle"))
    assert result.counterfactual.labels[LABELS.index("bicycle")] == "motorcycle"
    assert result.steering[LABELS.index("bicycle")].get("attribute") is None


# ---------------------------------------------------------------- bare "what if" (no question)
def test_effect_report_lists_dependents(reasoner, world):
    # pick a node that has children in the learned graph
    parent = next(j for j in range(len(LABELS)) if (world.adjacency[:, j] > 0).any())
    result = reasoner.intervene(world, Intervention("remove", nodes=[parent]))
    effect = reasoner.effect_report(result)
    assert effect["affected_objects"], "a node with children must affect them"
    assert all(item["relative_change"] > 0 for item in effect["affected_objects"])
    assert any(item["direct"] for item in effect["affected_objects"])
    assert "influence" in effect["conclusion"]
    assert LABELS[parent] not in effect["scene_after"] or LABELS.count(LABELS[parent]) > 1


def test_effect_report_without_dependents(reasoner, world):
    leaf = next((j for j in range(len(LABELS)) if not (world.adjacency[:, j] > 0).any()), None)
    if leaf is None:
        pytest.skip("every node has children in this random graph")
    effect = reasoner.effect_report(reasoner.intervene(world, Intervention("remove", nodes=[leaf])))
    assert effect["affected_objects"] == []
    assert "nothing else in the scene changes" in effect["conclusion"]


# ---------------------------------------------------------------- recompute mode (scene edit)
def _normal_pipeline(reasoner, raw, world, boxes, class_ids):
    """What a normal question does: encode nodes (embedding + class + box), build the graph,
    propagate, decode."""
    out = reasoner.model.forward_features(raw.unsqueeze(0), world.question_embedding, boxes.unsqueeze(0),
                                          class_ids.unsqueeze(0))
    return out["logits"][0], out["adjacency"][0]


def test_recompute_replace_answers_like_a_normal_question(reasoner, world):
    reasoner.mode = "recompute"
    result = reasoner.intervene(world, Intervention("replace", target="umbrella", new_label="dog"))
    assert result.mode == "recompute"
    assert result.counterfactual.class_ids[LABELS.index("umbrella")] == label_to_id("dog")
    logits, adjacency = _normal_pipeline(reasoner, result.counterfactual.raw_objects, world, BOXES,
                                         result.counterfactual.class_ids)
    assert torch.allclose(result.counterfactual.logits, logits, atol=1e-5)
    assert torch.allclose(result.counterfactual.adjacency, adjacency, atol=1e-6)


def test_recompute_remove_answers_like_a_normal_question(reasoner, world):
    reasoner.mode = "recompute"
    result = reasoner.intervene(world, Intervention("remove", target="car"))
    keep = [i for i, label in enumerate(LABELS) if label != "car"]
    logits, adjacency = _normal_pipeline(reasoner, world.raw_objects[keep], world, BOXES[keep],
                                         world.class_ids[keep])
    assert torch.allclose(result.counterfactual.logits, logits, atol=1e-5)
    assert torch.allclose(result.counterfactual.adjacency, adjacency, atol=1e-6)


def test_recompute_without_change_reproduces_the_real_scene(reasoner, world):
    reasoner.mode = "recompute"
    result = reasoner._finish_recompute(world, Intervention("replace", target="umbrella", new_label="umbrella"),
                                        world.objects.clone(), list(world.labels), world.raw_objects.clone(),
                                        [], {i: i for i in range(len(LABELS))}, {})
    assert torch.allclose(result.counterfactual.logits, world.logits, atol=1e-5)
    rel = reasoner.relationship_changes(result)
    assert rel["new"] == [] and rel["lost"] == [] and rel["changed"] == []


def test_relationship_changes_are_consistent(reasoner, world):
    reasoner.mode = "recompute"
    result = reasoner.intervene(world, Intervention("replace", target="bicycle", new_label="motorcycle"))
    rel = reasoner.relationship_changes(result)
    edges_before = int((world.adjacency > 1e-6).sum())
    edges_after = int((result.counterfactual.adjacency > 1e-6).sum())
    # every old edge is either lost or kept; every new-graph edge is either new or kept
    kept = edges_before - len(rel["lost"])
    assert kept == edges_after - len(rel["new"])
    assert kept == rel["unchanged"] + len(rel["changed"])
    focus = rel["focus"][0]
    assert focus["before_name"] == "bicycle(1)" and focus["after_name"] == "motorcycle(1)"
    assert len(focus["after_top_scores"]) == 3


def test_recompute_add_and_remove_focus(reasoner, world):
    reasoner.mode = "recompute"
    added = reasoner.relationship_changes(reasoner.intervene(world, Intervention("add", new_label="dog")))
    assert added["focus"][0]["after_name"] == "dog(new)" and "before_name" not in added["focus"][0]
    removed = reasoner.relationship_changes(reasoner.intervene(world, Intervention("remove", target="umbrella")))
    assert "after_name" not in removed["focus"][0]
    assert all(item["reason"] == "object removed" for item in removed["lost"] if "umbrella(2)" in item["edge"])


# ---------------------------------------------------------------- relation graph (soft + geometry)
def test_soft_graph_never_empty_and_rows_sum_to_one(reasoner, world):
    rows = world.adjacency.sum(dim=-1)
    assert torch.allclose(rows, torch.ones_like(rows), atol=1e-5)
    assert ((world.adjacency > 0).sum(dim=-1) == 3).all()       # top-3 parents per object


def test_geometry_changes_relations(reasoner, world):
    moved = BOXES.clone()
    moved[1] = torch.tensor([0.85, 0.85, 0.99, 0.99])           # move the bicycle far away
    other = reasoner.observe_from_tensors(world.raw_objects, LABELS, world.question, moved)
    assert not torch.allclose(other.edge_probs, world.edge_probs)


def test_contact_targets_and_box_features():
    from models.scm_graph import pairwise_box_features
    from train import contact_targets
    target = contact_targets(BOXES)
    assert target[0, 1] == 1 and target[0, 2] == 1               # person touches bicycle and umbrella
    assert target[0, 5] == 0 and target[3, 4] == 0                # far apart
    feats = pairwise_box_features(BOXES.unsqueeze(0))[0]
    assert feats.shape == (6, 6, 9)
    assert feats[0, 1, 2] == 0 and feats[0, 5, 2] > 0              # horizontal gap
