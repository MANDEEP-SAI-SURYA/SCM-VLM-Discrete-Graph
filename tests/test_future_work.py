"""Adaptive loss weighting + curriculum (losses/balancing.py) and neuro-symbolic knowledge
(models/knowledge.py, models/symbolic.py).  Stubbed DETR / BERT only."""

import math

import pytest

torch = pytest.importorskip("torch")
nn = torch.nn

import models.scm_vlm as scm_vlm_module
from configs.coco_labels import label_to_id
from losses.balancing import LossBalancer, competence, select_by_competence, stage_multiplier
from models.knowledge import (KNOWLEDGE_RELATIONS, NUM_KNOWLEDGE_RELATIONS, build_knowledge_base,
                              knowledge_triples, load_external_triples)
from models.symbolic import SymbolicAnswerer, edited_scene

LABELS = ["person", "bicycle", "umbrella", "car", "dog", "bus"]
BOXES = torch.tensor([
    [0.40, 0.30, 0.52, 0.80], [0.38, 0.55, 0.56, 0.90], [0.39, 0.20, 0.54, 0.35],
    [0.00, 0.50, 0.20, 0.65], [0.70, 0.80, 0.80, 0.95], [0.85, 0.05, 0.99, 0.30],
])
CLASS_IDS = torch.tensor([label_to_id(label) for label in LABELS])
VOCAB = {0: "yes", 1: "no", 2: "bicycle", 3: "motorcycle", 4: "0", 5: "1", 6: "2", 7: "red", 8: "dog", 9: "cat"}
ANSWER2IDX = {v: k for k, v in VOCAB.items()}


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
        model = scm_vlm_module.SCMVLM(num_classes=len(VOCAB), **kwargs)
        model.set_answer_vocab(VOCAB)
        return model

    return make


def _detections():
    return {"embeddings": torch.randn(len(LABELS), 256, generator=torch.Generator().manual_seed(1)),
            "boxes": BOXES, "labels": LABELS, "label_ids": CLASS_IDS}


def _ground_truth():
    """Tiny synthetic COCO: person always on top of / touching a bicycle; dogs far away."""
    gt = {}
    for image in range(30):
        gt[image] = [
            {"label": "person", "box": [0.40, 0.20, 0.55, 0.60], "iscrowd": 0},
            {"label": "bicycle", "box": [0.38, 0.55, 0.58, 0.90], "iscrowd": 0},
        ]
        if image % 3 == 0:
            gt[image].append({"label": "dog", "box": [0.85, 0.85, 0.95, 0.95], "iscrowd": 0})
    gt[999] = [{"label": "person", "box": [0.0, 0.0, 0.1, 0.1], "iscrowd": 0},
               {"label": "dog", "box": [0.05, 0.05, 0.12, 0.12], "iscrowd": 0}]
    return gt


# ===================================================================== loss balancing
def test_fixed_balancer_reproduces_the_previous_weighted_sum():
    base = {"task": 1.0, "cf": 1.0, "edge": 0.5, "acyclic": 0.05, "sparse": 0.01, "entropy": 0.05}
    terms = {k: torch.tensor(v) for k, v in
             {"task": 2.0, "cf": 1.5, "edge": 0.4, "acyclic": 0.2, "sparse": 0.3, "entropy": 0.1}.items()}
    total, weights = LossBalancer(base)(terms)
    expected = 2.0 + 1.5 + 0.5 * 0.4 + 0.05 * 0.2 + 0.01 * 0.3 + 0.05 * 0.1
    assert float(total) == pytest.approx(expected)
    assert weights["edge"] == 0.5


def test_uncertainty_weights_are_learned_and_downweight_noisy_objectives():
    balancer = LossBalancer({"task": 1.0, "cf": 1.0, "edge": 1.0}, method="uncertainty")
    optimizer = torch.optim.SGD(balancer.parameters(), lr=0.1)
    for _ in range(200):           # cf is a much larger (noisier) loss than task
        total, _ = balancer({"task": torch.tensor(0.1), "cf": torch.tensor(3.0), "edge": torch.tensor(0.5)})
        optimizer.zero_grad()
        total.backward()
        optimizer.step()
    # optimum of exp(-s) L + s/2 is s = log(2L): weight 1/(2L)
    assert math.exp(-float(balancer.log_vars["cf"])) == pytest.approx(1 / 6.0, rel=0.05)
    assert math.exp(-float(balancer.log_vars["task"])) > math.exp(-float(balancer.log_vars["cf"]))


def test_dwa_gives_more_weight_to_the_objective_that_stopped_improving():
    balancer = LossBalancer({"task": 1.0, "cf": 1.0, "edge": 1.0}, method="dwa")
    for epoch, (task, cf) in enumerate([(2.0, 2.0), (1.0, 2.0)]):   # task improves, cf plateaus
        balancer.set_epoch(epoch, 10)
        balancer({"task": torch.tensor(task), "cf": torch.tensor(cf), "edge": torch.tensor(1.0)})
        balancer.end_epoch()
    balancer.set_epoch(2, 10)
    assert balancer.dwa_weights["cf"] > balancer.dwa_weights["task"]
    assert sum(balancer.dwa_weights.values()) == pytest.approx(3.0)


def test_staged_regularisation_schedule():
    assert [stage_multiplier(e, 10, 0.2, 0.3) for e in range(10)] == pytest.approx(
        [0, 0, 1 / 3, 2 / 3, 1, 1, 1, 1, 1, 1])
    balancer = LossBalancer({"task": 1.0, "cf": 1.0, "edge": 0.5, "acyclic": 0.05}, curriculum=True)
    balancer.set_epoch(0, 10)
    assert not balancer.uses("cf") and balancer.uses("task") and balancer.uses("edge")
    total, weights = balancer({"task": torch.tensor(1.0), "acyclic": torch.tensor(5.0), "edge": torch.tensor(1.0)})
    assert float(total) == pytest.approx(1.5) and weights["acyclic"] == 0.0
    balancer.set_epoch(9, 10)
    assert balancer.uses("cf")


def test_competence_curriculum_starts_easy_and_ends_with_all_data():
    assert competence(0, 10, 0.3) == pytest.approx(0.3)
    assert competence(5, 10, 0.3) == pytest.approx(1.0)
    levels = [competence(e, 10, 0.3) for e in range(10)]
    assert levels == sorted(levels)
    difficulty = {i: float(i) for i in range(10)}
    assert select_by_competence(list(range(10))[::-1], difficulty, 0.3) == [2, 1, 0]
    assert select_by_competence([3, 1], difficulty, 1.0) == [3, 1]


def test_question_difficulty_orders_question_types():
    import train

    easy = train.question_difficulty({"question": "Is it sunny?", "answer_type": "yes/no"})
    hard = train.question_difficulty({"question": "How many people are standing next to the bus?",
                                      "answer_type": "number"})
    assert easy < hard


# ===================================================================== knowledge base
def test_knowledge_base_from_annotations():
    knowledge = build_knowledge_base(_ground_truth(), exclude_image_ids={999})
    person, bicycle, dog = label_to_id("person"), label_to_id("bicycle"), label_to_id("dog")
    rel = {name: i for i, name in enumerate(KNOWLEDGE_RELATIONS)}
    assert knowledge.shape[-1] == NUM_KNOWLEDGE_RELATIONS
    assert knowledge[person, bicycle, rel["touches"]] > 0.8
    assert knowledge[person, bicycle, rel["above"]] > 0.8 and knowledge[bicycle, person, rel["below"]] > 0.8
    assert knowledge[person, dog, rel["touches"]] < 0.4          # never touch (smoothed towards the global rate)
    # the excluded (validation) image, where person and dog touch, did not leak in
    with_val = build_knowledge_base(_ground_truth())
    assert with_val[person, dog, rel["touches"]] > knowledge[person, dog, rel["touches"]]
    triples = knowledge_triples(knowledge, 0.8)
    assert ("person", "above", "bicycle") in [t[:3] for t in triples]


def test_external_triples(tmp_path):
    path = tmp_path / "kb.tsv"
    path.write_text("# comment\nperson\trides\tbicycle\t0.9\nunicorn\trides\tbicycle\t1.0\ndog\tchases\tcat\n")
    knowledge, added = load_external_triples(path, torch.zeros(92, 92, NUM_KNOWLEDGE_RELATIONS))
    assert added == 2
    assert knowledge[label_to_id("person"), label_to_id("bicycle"), -1] == pytest.approx(0.9)
    assert knowledge[label_to_id("dog"), label_to_id("cat"), -1] == pytest.approx(1.0)


@pytest.mark.parametrize("graph_mode", ["soft", "discrete"])
def test_knowledge_prior_changes_edge_scores_and_gets_gradients(make_model, graph_mode):
    model = make_model(graph_mode=graph_mode, knowledge_prior=True).train()
    model.set_knowledge(build_knowledge_base(_ground_truth()))
    q = torch.randn(1, 768)
    raw = _detections()["embeddings"].unsqueeze(0)
    with_ids = model.forward_features(raw, q, BOXES.unsqueeze(0), CLASS_IDS.unsqueeze(0))
    prior = model.graph.knowledge_prior(CLASS_IDS.unsqueeze(0))
    assert prior.shape == (1, len(LABELS), len(LABELS))
    assert prior.abs().sum() > 0
    with_ids["logits"].sum().backward()
    assert model.graph.knowledge_prior.gate.grad is not None
    assert model.graph.knowledge_prior.mlp[0].weight.grad.abs().sum() > 0


def test_knowledge_buffer_is_saved_and_reloaded(make_model, tmp_path):
    import train
    from models.scm_vlm import build_model_from_checkpoint

    model = make_model(graph_mode="discrete", knowledge_prior=True, symbolic_fusion=True)
    knowledge = build_knowledge_base(_ground_truth())
    model.set_knowledge(knowledge)
    dataset = type("D", (), {"answer2idx": ANSWER2IDX, "idx2answer": VOCAB})()
    path = tmp_path / "ckpt.pt"
    train.save_checkpoint(model, dataset, 1, 0.5, 30.0, str(path))
    checkpoint = torch.load(path, weights_only=False)
    assert checkpoint["architecture"]["knowledge_prior"] and checkpoint["architecture"]["symbolic_fusion"]
    rebuilt = build_model_from_checkpoint(checkpoint)
    result = rebuilt.load_state_dict(checkpoint["trainable_state_dict"], strict=False)
    assert not result.unexpected_keys
    assert not [k for k in result.missing_keys if not k.startswith(("vision.", "text."))]
    assert torch.equal(rebuilt.graph.knowledge_prior.knowledge, knowledge)


def test_old_checkpoints_still_build_without_the_new_modules(make_model):
    from models.scm_vlm import build_model_from_checkpoint

    model = build_model_from_checkpoint({"num_classes": 10, "architecture": {"graph_mode": "soft"}})
    assert model.graph.knowledge_prior is None and not model.symbolic_fusion


# ===================================================================== symbolic fusion
def test_symbolic_answerer_on_the_scene():
    answerer = SymbolicAnswerer(VOCAB)
    boxes = BOXES.tolist()
    assert answerer.answer("Is there a dog?", LABELS, boxes).answer == "yes"
    assert answerer.answer("How many dogs are there?", LABELS, boxes).answer == "1"
    evidence = answerer.evidence("Is there a cat?", LABELS, boxes)
    assert evidence[0, ANSWER2IDX["no"]] == 1 and evidence.sum() == 1
    assert answerer.evidence("Why is the man happy?", LABELS, boxes).sum() == 0     # rules do not apply


def test_edited_scene():
    labels, boxes = edited_scene(LABELS, BOXES, "remove", [4])
    assert "dog" not in labels and len(boxes) == 5
    labels, _ = edited_scene(LABELS, BOXES, "replace", [1], "motorcycle")
    assert labels[1] == "motorcycle"


def test_symbolic_fusion_follows_the_intervention(make_model):
    """Removing the only dog must flip the symbolic evidence of 'Is there a dog?' to 'no'."""
    import train

    model = make_model(graph_mode="discrete", symbolic_fusion=True).eval()
    detections = _detections()
    question = "Is there a dog?"
    out = train.forward_detections(model, detections, torch.randn(768), question)
    assert out["symbolic_prior"][0, ANSWER2IDX["yes"]] == 1
    gate = float(model.symbolic_gate.gate_value(out["question_embedding"]))
    assert torch.allclose(out["logits"] - out["neural_logits"], gate * out["symbolic_prior"])
    prior = model.counterfactual_evidence(question, detections["labels"], detections["boxes"], "remove", [4])
    assert prior[0, ANSWER2IDX["no"]] == 1
    cf = model.intervene(out, "remove", [4], symbolic_prior=prior)
    assert torch.allclose(cf["logits"] - cf["neural_logits"], gate * prior)


def test_whatif_reasoner_uses_symbolic_fusion(make_model):
    from models.do_intervention import DoInterventionReasoner, Intervention

    model = make_model(graph_mode="discrete", symbolic_fusion=True, knowledge_prior=True).eval()
    with torch.no_grad():
        model.symbolic_gate.gate.bias.fill_(30.0)        # trust the rules completely
    reasoner = DoInterventionReasoner(model, VOCAB, steerer=None, mode="recompute")
    world = reasoner.observe_from_tensors(torch.randn(len(LABELS), 256), LABELS, "Is there a dog?", BOXES)
    assert VOCAB[int(world.logits.argmax())] == "yes"
    result = reasoner.intervene(world, Intervention("remove", target="dog"))
    assert VOCAB[int(result.counterfactual.logits.argmax())] == "no"


@pytest.mark.parametrize("weighting,curriculum", [("uncertainty", True), ("dwa", False)])
def test_training_step_with_all_new_features(make_model, weighting, curriculum):
    import train

    model = make_model(graph_mode="discrete", knowledge_prior=True, symbolic_fusion=True).train()
    model.set_knowledge(build_knowledge_base(_ground_truth()))
    settings = {"cf_mode": "supervised", "lambda_cf": 1.0, "lambda_edge": 0.5, "irrelevant_graph_check": True}
    settings["balancer"] = train.make_balancer(settings, "discrete", weighting, curriculum)
    settings["balancer"].set_epoch(9, 10)
    detections = _detections()
    q = torch.randn(768)
    loss, _, parts = train.factual_step(model, detections, q, torch.tensor(5), settings, "How many dogs are there?")
    example = {"type": "existence", "kind": "remove", "nodes": [4], "new_label": None, "new_label_id": None,
               "question": "Is there a dog?", "factual_answer": "yes", "cf_answer": "no", "mentioned": ["dog"]}
    cf_loss, _, cf_parts = train.counterfactual_step(model, detections, q, example, ANSWER2IDX, settings)
    total = loss + cf_loss
    assert torch.isfinite(total)
    total.backward()
    assert model.symbolic_gate.gate.weight.grad is not None
    assert model.graph.knowledge_prior.mlp[0].weight.grad.abs().sum() > 0
    if weighting == "uncertainty":
        assert settings["balancer"].log_vars["cf"].grad is not None
    assert "weights" in parts and "cf" in cf_parts["weights"]


def test_counterfactual_items_are_skipped_during_warmup(make_model):
    import train

    model = make_model(graph_mode="soft").train()
    settings = {"cf_mode": "supervised", "lambda_cf": 1.0, "lambda_edge": 0.5, "irrelevant_graph_check": False}
    settings["balancer"] = train.make_balancer(settings, "soft", "fixed", curriculum=True)
    settings["balancer"].set_epoch(0, 10)
    example = {"type": "existence", "kind": "remove", "nodes": [4], "question": "Is there a dog?",
               "factual_answer": "yes", "cf_answer": "no", "mentioned": ["dog"]}
    _, correct, parts = train.counterfactual_step(model, _detections(), torch.randn(768), example,
                                                  ANSWER2IDX, settings)
    assert correct is None and parts["skipped"]
