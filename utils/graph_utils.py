import networkx as nx
import numpy as np
import torch


def adjacency_to_graph(adjacency: torch.Tensor | np.ndarray, threshold: float = 0.5) -> nx.DiGraph:
    if isinstance(adjacency, torch.Tensor):
        adjacency = adjacency.detach().cpu().numpy()
    if adjacency.ndim == 3:
        adjacency = adjacency[0]

    graph = nx.DiGraph()
    n = adjacency.shape[0]
    graph.add_nodes_from(range(n))

    for i in range(n):
        for j in range(n):
            if i != j and adjacency[i, j] > threshold:
                graph.add_edge(i, j, weight=float(adjacency[i, j]))

    return graph
