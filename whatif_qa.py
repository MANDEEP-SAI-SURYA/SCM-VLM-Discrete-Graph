"""Answer "what if" (counterfactual) questions with do() interventions on SCM-VLM.

Works with any image: pass its path with --image.

Usage examples:
    python whatif_qa.py --image photo.jpg --whatif "What if there were no umbrella, what is the person holding?"
    python whatif_qa.py --image street.png --whatif "What if the bicycle were a motorcycle? What is the man riding?"
    python whatif_qa.py --image room.jpg --whatif "What if the sofa were removed, how many chairs are there?"
    python whatif_qa.py --image park.jpg --whatif "What if there was a dog?" --question "Is there a dog?"
    python whatif_qa.py --image road.jpg --whatif "What if the traffic light turns green"
    python whatif_qa.py --image road.jpg --question "How many cars are there?" --do "remove:car"
    python whatif_qa.py --image road.jpg --question "What is he riding?" --do "replace:#3->motorcycle"
    python whatif_qa.py --image road.jpg --whatif "What if the cars were removed, is the road busy?" --use-llm

Each run prints the answer in the real scene and in the intervened scene,
the nodes that were intervened on, their descendants (the only nodes that
are recomputed), and saves a before/after causal-graph figure.

Two answers are reported:
* scene-graph answer: read off the (counterfactual) scene graph - objects,
  boxes and intervened attributes (counting, existence, colour, relations).
* SCM-VLM classifier answer: the trained answer head on the causal node states.
"""

from __future__ import annotations

import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import json
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import torch

from models.do_intervention import (
    CounterfactualResult,
    DoInterventionReasoner,
    parse_do_spec,
    parse_whatif,
)
from models.scene_answer import answer_from_scene

# The causal graph is question-conditioned.  When the user asks a bare
# "what if ...?" there is no question, so the graph is built with this neutral
# one and the answer is the *effect* of the intervention, not a VQA answer.
NEUTRAL_QUESTION = "What is happening in this scene?"


# =====================================================================
# Visualisation: factual graph vs. intervened graph
# =====================================================================

def _draw(axis, adjacency, names, colours, title, positions):
    graph = nx.DiGraph()
    for i, name in enumerate(names):
        graph.add_node(i, label=name)
    n = adjacency.shape[0]
    for i in range(n):
        for j in range(n):
            if i != j and adjacency[i, j] > 1e-6:
                graph.add_edge(j, i, weight=float(adjacency[i, j]))   # A[i, j]: j -> i
    widths = [0.6 + 2.5 * graph[u][v]["weight"] for u, v in graph.edges]
    nx.draw_networkx_nodes(graph, positions, node_color=colours, node_size=900,
                           edgecolors="black", linewidths=1.0, ax=axis)
    nx.draw_networkx_edges(graph, positions, width=widths, arrows=True, arrowsize=14,
                           edge_color="#555555", connectionstyle="arc3,rad=0.08",
                           node_size=900, ax=axis)
    nx.draw_networkx_labels(graph, positions, labels=nx.get_node_attributes(graph, "label"),
                            font_size=7, ax=axis)
    axis.set_title(title, fontsize=10)
    axis.set_xlim(-1.25, 1.25)
    axis.set_ylim(-1.25, 1.25)
    axis.set_aspect("equal")
    axis.axis("off")


def save_comparison(result: CounterfactualResult, path: Path) -> None:
    factual, cf = result.factual, result.counterfactual
    n_f, n_c = len(factual.labels), len(cf.labels)
    layout = nx.circular_layout(range(max(n_f, n_c)))
    inv_map = {c: f for f, c in result.cf_index_map.items()}

    f_colours = []
    for i in range(n_f):
        if i in result.intervened_nodes:
            f_colours.append("#d62728")
        elif i in result.descendants:
            f_colours.append("#ff9f40")
        else:
            f_colours.append("#c7d7ea")
    c_colours, c_positions, c_names = [], {}, []
    for c in range(n_c):
        f = inv_map.get(c)
        c_positions[c] = layout[f if f is not None else c]
        # keep the factual node ids so both panels can be compared directly
        c_names.append(f"{f}:{cf.labels[c]}" if f is not None else f"new:{cf.labels[c]}")
        if f is None or f in result.intervened_nodes:
            c_colours.append("#d62728")
        elif f in result.descendants:
            c_colours.append("#ff9f40")
        else:
            c_colours.append("#c7d7ea")

    figure, axes = plt.subplots(1, 2, figsize=(14, 7))
    _draw(axes[0], factual.adjacency.cpu(), [f"{i}:{name}" for i, name in enumerate(factual.labels)], f_colours,
          "Real scene (learned relations)", {i: layout[i] for i in range(n_f)})
    _draw(axes[1], cf.adjacency.cpu(), c_names, c_colours,
          f"After {result.intervention.describe(result.mode)}"
          + ("  (relations re-computed)" if result.mode == "recompute" else ""), c_positions)
    if result.mode == "recompute":
        figure.suptitle("red = changed object   orange = objects whose state changed   "
                        "blue = unchanged", fontsize=9)
    else:
        figure.suptitle("red = intervened node(s)   orange = descendants (recomputed)   "
                        "blue = unaffected (factual value kept)", fontsize=9)
    path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(path, bbox_inches="tight", dpi=160)
    plt.close(figure)


# =====================================================================
# Optional natural-language answer through the configured LLM
# =====================================================================

def llm_effect_answer(whatif_text: str, effect: dict, model_name: str) -> str:
    """Natural-language consequences of a bare what-if, grounded in the scene and
    the learned causal graph; real-world common sense must be labelled as such."""
    from scene_graph_qa import load_project_env

    load_project_env()
    key = os.getenv("GROQ_API_KEY")
    if not key:
        raise RuntimeError("GROQ_API_KEY is not set (put it in .env).")
    from openai import OpenAI

    prompt = (
        "A vision-language model with a structural causal graph over the detected objects was asked a "
        "counterfactual question about an image.\n"
        f"Question: {whatif_text}\n"
        f"Real scene: {effect['scene_before']}\n"
        f"Intervention: {effect['scene_change']}\n"
        f"Counterfactual scene: {effect['scene_after']}\n"
        f"Causal edges cut: {'; '.join(effect['edges_cut']) or 'none'}\n"
        f"Model's causal-graph result: {effect['conclusion']}\n\n"
        "In 2-3 sentences, say what would change in this scene. First state what the model's causal graph "
        "says. Then give the likely real-world consequences, explicitly labelled 'Common sense:'. Only refer "
        "to objects that are in the scene."
    )
    client = OpenAI(api_key=key, base_url="https://api.groq.com/openai/v1")
    return client.responses.create(model=model_name, input=prompt).output_text.strip()


def llm_whatif_answer(whatif_text: str, summary: dict, factual_labels, cf_labels, model_name: str,
                      scene_lines: str = "") -> str:
    from scene_graph_qa import llm_answer, load_project_env

    load_project_env()
    evidence = (
        f"Objects detected in the real image: {', '.join(factual_labels)}\n"
        f"Intervention applied: {summary['intervention']}\n"
        f"Objects in the counterfactual scene: {', '.join(cf_labels)}\n"
        f"Objects whose representation changed (descendants): {', '.join(summary['descendants']) or 'none'}\n"
        f"SCM-VLM factual answer: {summary['factual_answer']} ({summary['factual_confidence']:.2f})\n"
        f"SCM-VLM counterfactual answer: {summary['counterfactual_answer']} "
        f"({summary['counterfactual_confidence']:.2f})\n"
        f"{scene_lines}"
        f"{summary.get('relationship_lines', '')}"
        "Reason about the counterfactual scene, not the real one."
    )
    return llm_answer(whatif_text, evidence, model_name)


def _links(items) -> str:
    return ", ".join(f"{name} {weight:.2f}" for name, weight in items) or "-"


def print_relationships(rel: dict, limit: int = 8) -> None:
    print("\n=========================")
    print("RELATIONSHIPS: REAL SCENE -> EDITED SCENE")
    print("=========================")
    print("(a -> b means a influences b; numbers are edge weights)")
    for entry in rel["focus"]:
        name_before = entry.get("before_name")
        name_after = entry.get("after_name")
        title = f"{name_before} -> {name_after}" if name_before and name_after else (name_before or name_after)
        status = "" if name_after else "  (removed)"
        if not name_before:
            status = "  (added)"
        print(f"\nObject {title}{status}")
        if name_before:
            print(f"  before: influenced by: {_links(entry['before_parents'])}")
            print(f"          influences   : {_links(entry['before_children'])}")
        if name_after:
            print(f"  after : influenced by: {_links(entry['after_parents'])}")
            print(f"          influences   : {_links(entry['after_children'])}")
        if entry.get("before_top_scores") or entry.get("after_top_scores"):
            print(f"  strongest edge scores (before threshold): "
                  f"before [{_links(entry.get('before_top_scores', []))}]  "
                  f"after [{_links(entry.get('after_top_scores', []))}]")

    print(f"\nNew relations     : {len(rel['new'])}")
    for item in rel["new"][:limit]:
        print(f"  + {item['edge']}  ({item['weight']:.2f})")
    print(f"Lost relations    : {len(rel['lost'])}")
    for item in rel["lost"][:limit]:
        print(f"  - {item['edge']}  ({item['weight']:.2f}, {item['reason']})")
    print(f"Changed strength  : {len(rel['changed'])}")
    for item in rel["changed"][:limit]:
        print(f"  ~ {item['edge']}  {item['before']:.2f} -> {item['after']:.2f}")
    print(f"Unchanged         : {rel['unchanged']}")
    if rel["state_changes"]:
        top = ", ".join(f"{x['node']} {100 * x['relative_change']:.0f}%" for x in rel["state_changes"][:limit])
        print(f"Objects whose state changed most: {top}")


def print_classifier(summary: dict, legacy: bool, mode: str = "recompute") -> None:
    how = ("same pipeline as a normal question, on the edited scene" if mode == "recompute"
           else "strict do(): fixed graph, only descendants recomputed")
    print(f"SCM-VLM answer ({how})"
          + (" - old checkpoint: answer head ignores the question, retrain" if legacy else "") + ":")
    print(f"  real scene      : {summary['factual_answer']} ({summary['factual_confidence']:.3f})")
    print(f"  edited scene    : {summary['counterfactual_answer']} ({summary['counterfactual_confidence']:.3f})")
    print(f"  answer changed  : {'YES' if summary['answer_changed'] else 'no'}   "
          f"effect on P(real answer) {summary['effect_on_factual_answer']:+.4f}   "
          f"total variation {summary['total_variation']:.4f}")
    print("  top real        :", ", ".join(f"{a} {p:.3f}" for a, p in summary["factual_top"]))
    print("  top edited      :", ", ".join(f"{a} {p:.3f}" for a, p in summary["counterfactual_top"]))


# =====================================================================
# Main
# =====================================================================

def build_reasoner():
    from main import load_trained_model

    model, idx2answer = load_trained_model()
    return DoInterventionReasoner(model, idx2answer)


def main() -> None:
    parser = argparse.ArgumentParser(description="Counterfactual what-if QA with do() interventions")
    parser.add_argument("--image", required=True, help="path to any image (jpg, png, ...)")
    parser.add_argument("--whatif", help='e.g. "What if there were no umbrella, what is the person holding?"')
    parser.add_argument("--question", help="question to answer in the intervened world")
    parser.add_argument("--do", dest="do_spec",
                        help="explicit intervention: remove:car | replace:umbrella->bicycle | "
                             "attribute:traffic light->green | add:dog | remove:#3")
    parser.add_argument("--mode", choices=["recompute", "do"], default="recompute",
                        help="recompute (default): edit the scene, let the model re-calculate all "
                             "relations and answer like a normal question.  do: strict SCM do() - "
                             "keep the learned graph, cut the parents of the changed object, "
                             "recompute only its descendants.")
    parser.add_argument("--strength", type=float, default=0.5,
                        help="size of an attribute intervention relative to the node norm (default 0.5)")
    parser.add_argument("--use-llm", action="store_true", help="also phrase the answer with the Groq LLM")
    parser.add_argument("--llm-model", default="openai/gpt-oss-120b")
    parser.add_argument("--out-dir", default="results/whatif")
    args = parser.parse_args()

    if not args.whatif and not args.do_spec:
        parser.error("give --whatif \"What if ...\" or --do <spec>")

    try:
        if args.do_spec:
            intervention = parse_do_spec(args.do_spec)
            question = args.question
        else:
            intervention, question = parse_whatif(args.whatif)
            question = args.question or question
    except ValueError as error:
        parser.error(str(error))
    question = question or intervention.default_question()
    effect_only = question is None
    if not Path(args.image).is_file():
        parser.error(f"image not found: {args.image}")

    reasoner = build_reasoner()
    reasoner.attribute_strength = args.strength
    reasoner.mode = args.mode
    try:
        world = reasoner.observe(args.image, question or NEUTRAL_QUESTION)
    except RuntimeError as error:
        print(f"\n{error}")
        return

    print("\n=========================")
    print("DETECTED OBJECTS (SCM nodes)")
    print("=========================")
    for i, label in enumerate(world.labels):
        parents = [j for j in range(len(world.labels)) if world.adjacency[i, j] > 1e-6]
        parent_text = ", ".join(f"{j}:{world.labels[j]}" for j in parents) or "-"
        print(f"Node {i:2d}: {label:<15} parents: {parent_text}")

    try:
        result = reasoner.intervene(world, intervention)
    except (ValueError, IndexError) as error:
        print(f"\nCannot apply {intervention.describe(args.mode)}: {error}")
        return
    summary = reasoner.summarise(result)
    effect = reasoner.effect_report(result)
    summary["effect"] = effect

    print("\n=========================")
    print("WHAT-IF ANALYSIS")
    print("=========================")
    print(f"Question          : {question or '(none - reporting the effect of the intervention)'}")
    print(f"Intervention      : {summary['intervention']}")
    print(f"Intervened nodes  : {', '.join(summary['intervened_nodes'])}")
    print(f"Mode              : {args.mode}" + (" (relations re-calculated on the edited scene)"
                                                   if args.mode == "recompute" else " (strict SCM do())"))
    if args.mode == "do":
        print(f"Descendants       : {', '.join(summary['descendants']) or 'none (no downstream effect)'}")
        print(f"Unaffected nodes  : {summary['unaffected_nodes']} (factual values kept, drift="
              f"{summary['non_descendant_max_drift']:.2e})")
    for node, info in summary["steering"].items():
        if info.get("attribute"):
            print(f"x' for node {node}  : '{info['label']}' (text-guided; similarity to the description "
                  f"{info['similarity_before']:.2f} -> {info['similarity_after']:.2f})")
        else:
            print(f"x' for node {node}  : DETR reads it as '{info['label']}' (p={info['prob']:.2f}"
                  f"{'' if info['converged'] else ', did not fully converge'})")

    # ---------------- effect of the intervention (always shown) ------------------
    print("\n=========================")
    print("EFFECT OF THE INTERVENTION")
    print("=========================")
    print(f"Change            : {effect['scene_change']}")
    print(f"Real scene        : {effect['scene_before']}")
    print(f"Intervened scene  : {effect['scene_after']}")
    if effect["edges_cut"]:
        print(f"Causal edges cut  : {'; '.join(effect['edges_cut'])}")
    print(f"Consequence       : {effect['conclusion']}")

    # ---------------- how the relations between objects changed ------------------
    relations = reasoner.relationship_changes(result)
    summary["relationships"] = relations
    summary["relationship_lines"] = (
        "Relations added after the edit: " + ("; ".join(x["edge"] for x in relations["new"][:8]) or "none") + "\n"
        + "Relations lost after the edit: " + ("; ".join(x["edge"] for x in relations["lost"][:8]) or "none") + "\n")
    print_relationships(relations)

    # ---------------- answer to the follow-up question (real vs. intervened) ------
    scene_factual = scene_cf = None
    if effect_only:
        if not args.use_llm:
            example = (args.whatif or "what if ...").rstrip("?. ")
            label = intervention.new_label if intervention.kind in {"replace", "add"} else effect["target_label"]
            plural_label = label if label.endswith("s") else label + "s"
            print("\nTip: add a question after a comma to get an answer in both scenes, e.g.\n"
                  f'  --whatif "{example}, how many {plural_label} are there?"\n'
                  "or add --use-llm for a natural-language explanation of the consequences.")
    else:
        resolve = reasoner.resolver(world)
        scene_factual = answer_from_scene(question, reasoner.scene_objects(world), resolve)
        scene_cf = answer_from_scene(question, reasoner.scene_objects(world, result), resolve)

        print("\n=========================")
        print(f"ANSWER: {question}")
        print("=========================")
        print_classifier(summary, not getattr(reasoner.model, "question_aware", True), args.mode)
        print("\nScene-graph check (read from the detected objects and boxes):")
        if scene_cf is not None:
            print(f"  real scene      : {scene_factual.answer:<14} ({scene_factual.reason})")
            print(f"  edited scene    : {scene_cf.answer:<14} ({scene_cf.reason})")
            print("  answer changed  :", "YES" if scene_factual.answer != scene_cf.answer else "no")
        else:
            print("  not available for this question type (supported: how many / is there / what color / "
                  "what is the X riding|holding|on|near ...)" + ("" if args.use_llm else
                                                               "; add --use-llm for a free-form answer"))
    summary["scene_graph_factual"] = vars(scene_factual) if scene_factual else None
    summary["scene_graph_counterfactual"] = vars(scene_cf) if scene_cf else None

    out_dir = Path(args.out_dir)
    slug = Path(args.image).stem + "_" + re.sub(r"[^a-z0-9]+", "_", summary["intervention"].lower()).strip("_")
    slug = re.sub(r"[^A-Za-z0-9_]+", "_", slug)
    figure_path = out_dir / f"{slug}_graph.png"
    save_comparison(result, figure_path)
    with (out_dir / f"{slug}.json").open("w", encoding="utf-8") as file:
        json.dump({"image": args.image, "question": question, **summary}, file, indent=2,
                  default=lambda value: str(value))
    print(f"\nSaved graph comparison: {figure_path}")
    print(f"Saved summary         : {out_dir / (slug + '.json')}")

    if args.use_llm and effect_only:
        print("\nLLM ANSWER")
        print(llm_effect_answer(args.whatif or summary["intervention"], effect, args.llm_model))
    elif args.use_llm:
        text = args.whatif or f"{summary['intervention']} - {question}"
        print("\nLLM ANSWER")
        scene_lines = ""
        if scene_cf is not None:
            scene_lines = (f"Scene-graph answer in the real scene: {scene_factual.answer} ({scene_factual.reason})\n"
                           f"Scene-graph answer in the counterfactual scene: {scene_cf.answer} ({scene_cf.reason})\n")
        print(llm_whatif_answer(text, summary, world.labels, result.counterfactual.labels, args.llm_model,
                                scene_lines))


if __name__ == "__main__":
    with torch.inference_mode():
        main()
