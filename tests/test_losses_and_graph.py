import pytest

torch = pytest.importorskip("torch")

from losses.losses import acyclicity_loss, total_scm_loss
from models.intervention import InterventionEngine, find_descendants
from models.scm_graph import SCMGraph


def test_acyclicity_loss_detects_cycle() -> None:
    empty = torch.zeros(1, 3, 3)
    cycle = torch.tensor([[[0.0, 1.0], [1.0, 0.0]]])

    assert acyclicity_loss(empty).item() == pytest.approx(0.0)
    assert acyclicity_loss(cycle).item() > 0.0


def test_descendants_follow_directed_edges() -> None:
    # chain 0 -> 1 -> 2 -> 3; A[i, j] > 0 means j -> i (messages = A @ h)
    adjacency = torch.tensor(
        [
            [0.0, 0.8, 0.0, 0.0],
            [0.0, 0.0, 0.9, 0.0],
            [0.0, 0.0, 0.0, 0.7],
            [0.0, 0.0, 0.0, 0.0],
        ]
    ).T
    assert find_descendants(adjacency, 0) == [1, 2, 3]
    assert find_descendants(adjacency, 3) == []


def test_intervention_cuts_incoming_edges() -> None:
    engine = InterventionEngine()
    objects = torch.randn(1, 4, 8)
    adjacency = torch.rand(1, 4, 4)
    replacement = torch.randn(1, 8)
    nodes = torch.tensor([2])

    new_objects, new_adjacency, descendants = engine.intervene(objects, adjacency, nodes, replacement)

    # A[i, j] > 0 means j -> i, so the parents of node 2 live in row 2
    assert torch.allclose(new_adjacency[0, 2, :], torch.zeros(4))
    assert torch.allclose(new_objects[0, 2], replacement[0])
    assert len(descendants) == 1


def test_scm_graph_returns_scores_and_sparse_normalized_adjacency() -> None:
    graph = SCMGraph(object_dim=8, question_dim=6, top_k=2, threshold=0.0)
    objects = torch.randn(2, 5, 8)
    question = torch.randn(2, 6)

    out = graph(objects, question)

    assert out["scores"].shape == (2, 5, 5)
    assert out["probs"].shape == (2, 5, 5)
    assert out["adjacency"].shape == (2, 5, 5)
    assert torch.allclose(torch.diagonal(out["adjacency"], dim1=-2, dim2=-1), torch.zeros(2, 5))
    assert torch.all((out["hard_adjacency"] > 0).sum(dim=-1) <= 2)
    row_sums = out["adjacency"].sum(dim=-1)
    assert torch.all((row_sums == 0) | torch.isclose(row_sums, torch.ones_like(row_sums)))


def test_total_loss_includes_counterfactual_terms() -> None:
    logits = torch.randn(2, 5, requires_grad=True)
    cf_logits = torch.randn(2, 5, requires_grad=True)
    labels = torch.tensor([1, 3])
    cf_labels = torch.tensor([2, 4])
    adjacency = torch.rand(2, 4, 4)
    stable = logits.detach() + 0.1

    total, parts = total_scm_loss(logits, labels, adjacency, cf_logits, cf_labels, stable)

    assert total.requires_grad
    assert parts["task"].item() > 0
    assert parts["counterfactual"].item() > 0
    assert parts["sparsity"].item() >= 0
