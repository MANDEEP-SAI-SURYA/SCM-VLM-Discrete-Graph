from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
import torch

from utils.graph_utils import adjacency_to_graph


def visualize_graph(adjacency: torch.Tensor, save_path: str | None = None, threshold: float = 0.00000001) -> None:
    graph = adjacency_to_graph(adjacency, threshold=threshold)
    plt.figure(figsize=(8, 8))
    pos = nx.spring_layout(graph, seed=7)
    weights = [graph[u][v]["weight"] * 2 for u, v in graph.edges]
    nx.draw(
        graph,
        pos,
        with_labels=True,
        node_size=500,
        arrows=True,
        width=weights,
    )

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()
    else:
        plt.show()
