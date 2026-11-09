"""Run question-guided scene-graph reasoning over an image.

Usage:
    python scene_graph_qa.py --image <any image> --question "What is the person riding?"
    python scene_graph_qa.py --image image.jpg --question "Why is he using an umbrella?" --use-llm
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import networkx as nx
from PIL import Image

from models.question_guided_scene_graph import QuestionGuidedSceneGraph
from models.text_encoder import TextEncoder
from models.vision_encoder import VisionEncoder


def load_project_env() -> None:
    """Load simple KEY=VALUE entries from the project's .env file."""
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.is_file():
        return
    for raw_line in env_path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


def evidence_text(nodes, edges) -> str:
    node_lines = [f"- {node.label} (relevance={node.relevance:.2f})" for node in nodes]
    node_names = {node.index: node.label for node in nodes}
    edge_lines = [
        f"- {node_names[edge.source]} {edge.relation} {node_names[edge.target]}"
        for edge in edges
    ]
    return "Relevant nodes:\n" + "\n".join(node_lines) + "\nRelationships:\n" + ("\n".join(edge_lines) or "- No relationship between the selected nodes.")


def fallback_answer(question: str, nodes, edges) -> str:
    """Transparent local fallback when no external LLM is configured."""
    node_names = ", ".join(node.label for node in nodes)
    if not nodes:
        return "I could not retrieve enough visual evidence to answer the question."
    relation = ""
    if edges:
        relation = f" The strongest retrieved relation is: {edges[0].relation}."
    return f"Retrieved visual evidence for '{question}': {node_names}.{relation} An LLM API key is required to turn this evidence into a natural-language answer."


def save_graph(image_path: str, nodes, edges, output_path: Path, title: str, show_edge_labels: bool) -> None:
    """Draw graph nodes at their detected image positions."""
    graph = nx.DiGraph()
    for node in nodes:
        graph.add_node(node.index, label=f"{node.index}: {node.label}", relevance=node.relevance)

    # Keep one direction per object pair. The complete graph remains legible
    # without duplicate arrows; the relevant graph still shows its relations.
    unique_edges = {}
    for edge in edges:
        pair = tuple(sorted((edge.source, edge.target)))
        unique_edges.setdefault(pair, edge)
    for edge in unique_edges.values():
        graph.add_edge(edge.source, edge.target, relation=edge.relation)

    with Image.open(image_path).convert("RGB") as image:
        width, height = image.size
        figure, axis = plt.subplots(figsize=(12, 8))
        axis.imshow(image)

    positions = {
        node.index: ((node.box[0] + node.box[2]) / 2, (node.box[1] + node.box[3]) / 2)
        for node in nodes
    }
    colors = [node.relevance for node in nodes]
    nx.draw_networkx_nodes(
        graph, positions, node_color=colors, cmap="YlOrRd", vmin=0, vmax=1,
        node_size=850, edgecolors="black", linewidths=1.0, ax=axis,
    )
    nx.draw_networkx_edges(
        graph, positions, edge_color="white", width=1.2, alpha=0.75,
        arrows=True, arrowsize=14, connectionstyle="arc3,rad=0.10", ax=axis,
    )
    nx.draw_networkx_labels(
        graph, positions,
        labels={index: data["label"] for index, data in graph.nodes(data=True)},
        font_size=8, font_color="black", ax=axis,
    )
    if show_edge_labels:
        nx.draw_networkx_edge_labels(
            graph, positions,
            edge_labels={(source, target): data["relation"] for source, target, data in graph.edges(data=True)},
            font_size=7, font_color="black", rotate=False, ax=axis,
        )

    axis.set_title(title)
    axis.set_xlim(0, width)
    axis.set_ylim(height, 0)
    axis.axis("off")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, bbox_inches="tight", dpi=180)
    plt.close(figure)


def llm_answer(question: str, evidence: str, model_name: str) -> str:
    """Generate an answer with Groq's OpenAI-compatible Responses API."""
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
        raise RuntimeError("GROQ_API_KEY is not set.")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise RuntimeError("Install the optional LLM dependency with: pip install openai") from error

    prompt = (
        "Answer the question using only the retrieved scene-graph evidence. "
        "Do not invent invisible causes or objects. If the evidence is insufficient, say so. "
        "Answer in one concise sentence.\n\n"
        f"Question: {question}\n\n{evidence}"
    )
    client = OpenAI(
        api_key=groq_key,
        base_url="https://api.groq.com/openai/v1",
    )
    response = client.responses.create(model=model_name, input=prompt)
    return response.output_text.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Question-guided scene graph QA")
    parser.add_argument("--image", required=True, help="path to any image")
    parser.add_argument("--question", required=True)
    parser.add_argument("--use-llm", action="store_true", help="Generate the final answer through the configured LLM API.")
    parser.add_argument("--llm-model", default="openai/gpt-oss-120b")
    parser.add_argument("--graph-dir", default="results/scene_graphs")
    arguments = parser.parse_args()
    load_project_env()

    reasoner = QuestionGuidedSceneGraph(VisionEncoder(freeze=True), TextEncoder(freeze=True))
    nodes, edges = reasoner.build(arguments.image, arguments.question)
    evidence_nodes, evidence_edges = reasoner.retrieve(nodes, edges)
    evidence = evidence_text(evidence_nodes, evidence_edges)
    graph_dir = Path(arguments.graph_dir)
    all_graph_path = graph_dir / "all_objects_graph.png"
    relevant_graph_path = graph_dir / "question_relevant_graph.png"
    save_graph(arguments.image, nodes, edges, all_graph_path, "All detected objects", show_edge_labels=False)
    save_graph(arguments.image, evidence_nodes, evidence_edges, relevant_graph_path, "Question-relevant subgraph", show_edge_labels=True)

    print("\nSCENE GRAPH NODES")
    for node in nodes:
        print(f"{node.index}: {node.label} | relevance={node.relevance:.2f}")
    print("\nQUESTION-RELEVANT EVIDENCE")
    print(evidence)
    print(f"\nSaved all-objects graph: {all_graph_path}")
    print(f"Saved question-relevant graph: {relevant_graph_path}")

    answer = llm_answer(arguments.question, evidence, arguments.llm_model) if arguments.use_llm else fallback_answer(arguments.question, evidence_nodes, evidence_edges)
    print("\nFINAL ANSWER")
    print(answer)


if __name__ == "__main__":
    main()
