import torch

# Graph convention (see models/causal_attention.py):  messages = A @ h, i.e.
# h_i <- sum_j A[i, j] h_j.  So A[i, j] > 0 means j -> i (j is a parent of i):
# row i = parents of i, column j = children of j.


def find_descendants(adjacency: torch.Tensor, node_index: int, threshold: float = 1e-8) -> list[int]:
    """Nodes reachable from ``node_index`` along j -> i edges (A[i, j] > threshold)."""
    graph = adjacency.detach().cpu()
    seen: set[int] = set()
    queue = [node_index]

    while queue:
        current = queue.pop(0)
        children = torch.nonzero(graph[:, current] > threshold, as_tuple=False).flatten().tolist()
        for child in children:
            if child != node_index and child not in seen:
                seen.add(child)
                queue.append(child)

    return sorted(seen)


class InterventionEngine:
    """Implements do(X_k = x'): cut incoming edges (row k) and replace the node value."""

    def intervene(
        self,
        objects: torch.Tensor,
        adjacency: torch.Tensor,
        node_indices: torch.Tensor,
        replacement_embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, list[list[int]]]:
        if objects.dim() == 2:
            objects = objects.unsqueeze(0)
        if adjacency.dim() == 2:
            adjacency = adjacency.unsqueeze(0)
        if node_indices.dim() == 0:
            node_indices = node_indices.unsqueeze(0)
        if replacement_embeddings.dim() == 1:
            replacement_embeddings = replacement_embeddings.unsqueeze(0)

        intervened_objects = objects.clone()
        intervened_adjacency = adjacency.clone()
        descendant_sets: list[list[int]] = []

        for batch_index in range(objects.shape[0]):
            node = int(node_indices[batch_index].item())
            if node < 0 or node >= objects.shape[1]:
                raise IndexError(f"node index {node} is outside the object range")

            # do(X_k = x') removes every parent -> k edge, i.e. row k.
            intervened_adjacency[batch_index, node, :] = 0.0
            intervened_objects[batch_index, node] = replacement_embeddings[batch_index]
            descendant_sets.append(find_descendants(intervened_adjacency[batch_index], node))

        return intervened_objects, intervened_adjacency, descendant_sets


def influenced_nodes(adjacency: torch.Tensor, sources: list[int], diagnostic: bool,
                     excluded: list[int] | None = None, threshold: float = 1e-8) -> list[int]:
    """Nodes whose state can change when ``sources`` change, in the (mutilated) graph.

    Causal messages only (parent -> child): the descendants of the sources.
    With diagnostic messages (child -> parent as well, Pearl-style belief propagation) evidence
    flows along edges in both directions, so every node connected to a source by an undirected
    path can change (e.g. co-parents of a common child: "explaining away").  ``excluded`` nodes
    (e.g. removed objects) are not traversed."""
    graph = adjacency.detach().cpu() > threshold
    if diagnostic:
        graph = graph | graph.T
    excluded = set(excluded or [])
    seen: set[int] = set()
    queue = [s for s in sources if s not in excluded]
    while queue:
        current = queue.pop(0)
        neighbours = torch.nonzero(graph[:, current], as_tuple=False).flatten().tolist()
        for node in neighbours:
            if node not in seen and node not in sources and node not in excluded:
                seen.add(node)
                queue.append(node)
    return sorted(seen)
