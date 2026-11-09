"""Held-out evaluation of a trained SCM-VLM (used at the end of train.py and by
relation_analysis.py).

1. Factual VQA accuracy: overall, by VQA answer type (yes/no, number, other) and for
   relational vs. non-relational questions (keyword heuristic, see RELATIONAL_WORDS).
2. Counterfactual accuracy on the held-out generated examples, per type
   (existence, counting, replacement, irrelevant) and subtype.
3. Graph knock-out on the trained model: accuracy and % of answers changed with
   (a) the learned graph, (b) message passing switched off, (c) a scrambled graph
   (every object's parents re-assigned to other objects), with exact McNemar p-values
   versus the learned graph.
4. Relation probes with bootstrap 95 % confidence intervals:
   - edge probability for touching vs. distant object pairs,
   - answer-head attention on the answer object (replacement examples; 1.0 = uniform),
   - sensitivity: % answers changed by relevant vs. irrelevant interventions.

Everything is written to results/ with a _<tag> suffix.
"""

from __future__ import annotations

import json
import math
import os
import random
import re

import numpy as np
import pandas as pd
import torch

from configs.config import DEVICE
from datasets.counterfactual import CF_TYPES

RELATIONAL_WORDS = (
    "on", "under", "above", "below", "behind", "beside", "next to", "near", "in front of",
    "on top of", "left", "right", "holding", "riding", "wearing", "carrying", "sitting",
    "standing", "lying", "inside", "between", "across", "over", "by", "attached", "hanging",
    "pulling", "pushing", "eating", "using", "playing", "touching", "against", "around",
)
_RELATIONAL = re.compile(r"\b(" + "|".join(re.escape(w) for w in sorted(RELATIONAL_WORDS, key=len, reverse=True)) + r")\b")


def is_relational(question: str) -> bool:
    """Heuristic: the question names a spatial relation or an interaction verb."""
    return bool(_RELATIONAL.search(question.lower()))


# =====================================================================
# statistics
# =====================================================================

def mcnemar_exact(correct_a, correct_b) -> dict:
    """Exact (binomial) McNemar test on paired correctness vectors."""
    a = np.asarray(correct_a, dtype=bool)
    b = np.asarray(correct_b, dtype=bool)
    only_a = int(np.sum(a & ~b))
    only_b = int(np.sum(~a & b))
    n = only_a + only_b
    if n == 0:
        return {"only_first_correct": 0, "only_second_correct": 0, "p_value": 1.0}
    k = min(only_a, only_b)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2 ** n
    return {"only_first_correct": only_a, "only_second_correct": only_b, "p_value": min(1.0, 2 * tail)}


def bootstrap_ci(values, statistic=np.mean, n_boot: int = 1000, seed: int = 0) -> dict:
    """Point estimate and percentile 95 % CI."""
    values = np.asarray(values, dtype=float)
    values = values[~np.isnan(values)]
    if len(values) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"), "n": 0}
    rng = np.random.default_rng(seed)
    stats = [statistic(values[rng.integers(0, len(values), len(values))]) for _ in range(n_boot)]
    return {"mean": float(statistic(values)), "ci_low": float(np.percentile(stats, 2.5)),
            "ci_high": float(np.percentile(stats, 97.5)), "n": int(len(values))}


def bootstrap_diff_ci(first, second, n_boot: int = 1000, seed: int = 0) -> dict:
    """mean(first) - mean(second) with a 95 % CI (independent resampling)."""
    a = np.asarray(first, dtype=float)
    b = np.asarray(second, dtype=float)
    if len(a) == 0 or len(b) == 0:
        return {"mean": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    rng = np.random.default_rng(seed)
    diffs = [a[rng.integers(0, len(a), len(a))].mean() - b[rng.integers(0, len(b), len(b))].mean()
             for _ in range(n_boot)]
    return {"mean": float(a.mean() - b.mean()), "ci_low": float(np.percentile(diffs, 2.5)),
            "ci_high": float(np.percentile(diffs, 97.5))}


# =====================================================================
# model passes
# =====================================================================

def _forward(model, detections, q_emb, **kwargs):
    return model.forward_features(
        detections["embeddings"].unsqueeze(0).to(DEVICE), q_emb.unsqueeze(0).to(DEVICE),
        detections["boxes"].unsqueeze(0).to(DEVICE), detections["label_ids"].unsqueeze(0).to(DEVICE), **kwargs)


def scramble_graph(adjacency: torch.Tensor, relation_probs, rng: random.Random):
    """Connect every object to the WRONG objects: permute the parent columns with a random
    derangement, keep each row's weights, drop self-loops and renormalise."""
    n = adjacency.shape[-1]
    if n < 2:
        return adjacency, relation_probs
    order = list(range(n))
    while True:
        rng.shuffle(order)
        if all(order[i] != i for i in range(n)):
            break
    index = torch.tensor(order, device=adjacency.device)
    scrambled = adjacency.index_select(-1, index).clone()
    scrambled[..., torch.arange(n), torch.arange(n)] = 0.0
    scrambled = scrambled / (scrambled.sum(dim=-1, keepdim=True) + 1e-8)
    if relation_probs is not None:
        relation_probs = relation_probs.index_select(2, index)
    return scrambled, relation_probs


@torch.no_grad()
def factual_predictions(model, dataset, indices, cache, seed: int = 0) -> pd.DataFrame:
    """Per-question predictions with the learned graph, with messages off and with a
    scrambled graph (graph variants only when the model has a graph)."""
    model.eval()
    rng = random.Random(seed)
    rows = []
    for index in indices:
        record = dataset.records[index]
        image_path, question, label = dataset[index]
        try:
            detections = cache.image(image_path)
            if len(detections["labels"]) == 0:
                continue
            q_emb = cache.question(question)
            out = _forward(model, detections, q_emb)
        except Exception as error:
            print(f"EVAL SKIP {image_path}: {error}")
            continue
        row = {
            "question_id": record.get("question_id"), "question": question,
            "answer": record["answer"], "answer_type": record.get("answer_type", "other"),
            "relational": is_relational(question), "num_objects": len(detections["labels"]),
            "pred_learned": int(out["logits"].argmax(-1)),
        }
        if model.graph is not None:
            off = _forward(model, detections, q_emb, messages=False)
            scrambled_a, scrambled_r = scramble_graph(out["adjacency"], out["relation_probs"], rng)
            scrambled = _forward(model, detections, q_emb, adjacency_override=scrambled_a,
                                 relation_probs_override=scrambled_r)
            row["pred_messages_off"] = int(off["logits"].argmax(-1))
            row["pred_scrambled"] = int(scrambled["logits"].argmax(-1))
        label = int(label)
        for key in [k for k in row if k.startswith("pred_")]:
            row["correct_" + key[5:]] = int(row[key] == label)
        rows.append(row)
    return pd.DataFrame(rows)


@torch.no_grad()
def counterfactual_predictions(model, examples, cache, answer2idx, irrelevant_graph_check=True) -> pd.DataFrame:
    """Per-example results on the held-out counterfactual examples."""
    from train import irrelevant_is_related

    model.eval()
    rows = []
    for example in examples:
        detections = cache.image(example["image_path"])
        out = _forward(model, detections, cache.question(example["question"]))
        related = (model.graph is not None and irrelevant_is_related(out, detections, example))
        new_raw = example["x_prime"].to(DEVICE) if example["kind"] == "replace" else None
        new_ids = [example["new_label_id"]] * len(example["nodes"]) if example["kind"] == "replace" else None
        cf = model.intervene(out, example["kind"], example["nodes"], new_raw=new_raw, new_class_ids=new_ids)
        factual_pred = int(out["logits"].argmax(-1))
        cf_pred = int(cf["logits"].argmax(-1))
        answer_attention = float("nan")
        if example.get("answer_node") is not None:
            n = out["answer_attention"].shape[-1]
            answer_attention = float(out["answer_attention"][0, example["answer_node"]]) * n   # 1.0 = uniform
        rows.append({
            "type": example["type"], "subtype": example["subtype"], "kind": example["kind"],
            "question": example["question"], "image_path": example["image_path"],
            "factual_answer": example["factual_answer"], "cf_answer": example["cf_answer"],
            "factual_correct": int(example["factual_answer"] in answer2idx and
                                   factual_pred == answer2idx[example["factual_answer"]]),
            "cf_correct": int(cf_pred == answer2idx[example["cf_answer"]]),
            "changed": int(cf_pred != factual_pred),
            "related_in_graph": int(related),
            "excluded": int(irrelevant_graph_check and related),
            "answer_attention_ratio": answer_attention,
        })
    return pd.DataFrame(rows)


@torch.no_grad()
def edge_probe(model, dataset, indices, cache, max_images: int = 2000) -> dict:
    """Mean edge probability per image for touching vs. distant object pairs."""
    from train import edge_supervision_loss

    if model.graph is None:
        return {}
    seen, contact, far = set(), [], []
    for index in indices:
        image_path, question, _ = dataset[index]
        if image_path in seen or len(seen) >= max_images:
            continue
        seen.add(image_path)
        detections = cache.image(image_path)
        if len(detections["labels"]) < 2:
            continue
        out = _forward(model, detections, cache.question(question))
        _, stats = edge_supervision_loss(out["edge_probs"][0], detections["boxes"].to(DEVICE))
        if stats["p_contact"] is not None:
            contact.append(float(stats["p_contact"]))
        if stats["p_far"] is not None:
            far.append(float(stats["p_far"]))
    return {"edge_prob_touching": bootstrap_ci(contact), "edge_prob_distant": bootstrap_ci(far),
            "touching_minus_distant": bootstrap_diff_ci(contact, far)}


# =====================================================================
# full evaluation
# =====================================================================

def _accuracy(series) -> float:
    return 100.0 * float(np.mean(series)) if len(series) else float("nan")


def run_full_evaluation(model, dataset, val_idx, cache, val_cf, tag=None, results_dir="results",
                        irrelevant_graph_check=True, seed: int = 0) -> dict:
    suffix = f"_{tag}" if tag else ""
    os.makedirs(results_dir, exist_ok=True)
    print("\n========================")
    print("FULL EVALUATION (held-out validation)")
    print("========================")

    factual = factual_predictions(model, dataset, val_idx, cache, seed)
    factual.to_csv(os.path.join(results_dir, f"eval_factual{suffix}.csv"), index=False)
    summary: dict = {"tag": tag, "architecture": model.architecture(), "num_questions": int(len(factual))}

    # 1. factual accuracy
    acc = {"overall": _accuracy(factual["correct_learned"])}
    for answer_type, group in factual.groupby("answer_type"):
        acc[f"answer_type={answer_type}"] = _accuracy(group["correct_learned"])
    for flag, group in factual.groupby("relational"):
        acc["relational" if flag else "non_relational"] = _accuracy(group["correct_learned"])
    acc["overall_ci"] = bootstrap_ci(factual["correct_learned"] * 100.0)
    summary["factual_accuracy"] = acc

    # 2. counterfactual accuracy
    cf_table = pd.DataFrame()
    if val_cf:
        cf_table = counterfactual_predictions(model, val_cf, cache, dataset.answer2idx, irrelevant_graph_check)
        cf_table.to_csv(os.path.join(results_dir, f"eval_counterfactual{suffix}.csv"), index=False)
        # Headline numbers use ALL held-out examples, so every model (also --graph-mode none)
        # is scored on the same set.  The graph-dependent "irrelevant" filter (node is not a
        # parent/child of a mentioned object in this model's learned graph) is reported separately.
        cf_acc = {"overall": _accuracy(cf_table["cf_correct"]), "num_examples": int(len(cf_table))}
        for kind in CF_TYPES:
            group = cf_table[cf_table["type"] == kind]
            cf_acc[kind] = _accuracy(group["cf_correct"])
            cf_acc[f"{kind}_n"] = int(len(group))
        for subtype, group in cf_table.groupby("subtype"):
            cf_acc[f"subtype={subtype}"] = _accuracy(group["cf_correct"])
        irrelevant = cf_table[cf_table["type"] == "irrelevant"]
        unrelated = irrelevant[irrelevant["related_in_graph"] == 0]
        cf_acc["irrelevant_unrelated_in_graph"] = _accuracy(unrelated["cf_correct"])
        cf_acc["irrelevant_unrelated_in_graph_n"] = int(len(unrelated))
        cf_acc["irrelevant_related_in_graph_n"] = int(len(irrelevant) - len(unrelated))
        summary["counterfactual_accuracy"] = cf_acc

    # 3. graph knock-out
    knockout = {}
    if "pred_messages_off" in factual:
        for variant in ("learned", "messages_off", "scrambled"):
            entry = {"accuracy": _accuracy(factual[f"correct_{variant}"])}
            if variant != "learned":
                entry["answers_changed_pct"] = 100.0 * float(np.mean(factual[f"pred_{variant}"] != factual["pred_learned"]))
                entry["mcnemar_vs_learned"] = mcnemar_exact(factual["correct_learned"], factual[f"correct_{variant}"])
                for flag, group in factual.groupby("relational"):
                    name = "relational" if flag else "non_relational"
                    entry[f"answers_changed_pct_{name}"] = 100.0 * float(
                        np.mean(group[f"pred_{variant}"] != group["pred_learned"]))
            knockout[variant] = entry
    else:
        knockout["note"] = "model has no graph (--graph-mode none): knock-outs not applicable"
    summary["graph_knockout"] = knockout

    # 4. relation probes
    probes = {"edges": edge_probe(model, dataset, val_idx, cache)}
    if len(cf_table):
        used = cf_table
        relevant = used[used["type"].isin(["counting", "replacement"]) | (used["subtype"] == "remove_target")]
        irrelevant = used[(used["type"] == "irrelevant") | (used["subtype"] == "remove_other")]
        probes["changed_pct_relevant"] = bootstrap_ci(relevant["changed"] * 100.0)
        probes["changed_pct_irrelevant"] = bootstrap_ci(irrelevant["changed"] * 100.0)
        probes["relevant_minus_irrelevant"] = bootstrap_diff_ci(relevant["changed"] * 100.0,
                                                                irrelevant["changed"] * 100.0)
        attention = used["answer_attention_ratio"].dropna()
        probes["answer_object_attention_ratio"] = bootstrap_ci(attention)
    summary["relation_probes"] = probes

    with open(os.path.join(results_dir, f"eval_summary{suffix}.json"), "w", encoding="utf-8") as file:
        json.dump(summary, file, indent=2, default=float)
    _print_summary(summary)
    print(f"\nWrote results/eval_summary{suffix}.json, eval_factual{suffix}.csv"
          + (f", eval_counterfactual{suffix}.csv" if len(cf_table) else ""))
    return summary


def _fmt_ci(entry) -> str:
    if not entry or entry.get("n", 1) == 0:
        return "n/a"
    return f"{entry['mean']:.3f} [{entry['ci_low']:.3f}, {entry['ci_high']:.3f}]"


def _print_summary(summary: dict) -> None:
    acc = summary["factual_accuracy"]
    print("Factual accuracy: " + ", ".join(f"{k} {v:.2f}" for k, v in acc.items() if not isinstance(v, dict)))
    if "counterfactual_accuracy" in summary:
        cf = summary["counterfactual_accuracy"]
        print("Counterfactual accuracy: " + ", ".join(
            f"{k} {v:.2f}" if isinstance(v, float) else f"{k} {v}" for k, v in cf.items()))
    for variant, entry in summary["graph_knockout"].items():
        if isinstance(entry, dict):
            extra = ""
            if "mcnemar_vs_learned" in entry:
                extra = (f", changed {entry['answers_changed_pct']:.1f}%, "
                         f"McNemar p={entry['mcnemar_vs_learned']['p_value']:.3g}")
            print(f"Knock-out {variant:<12}: accuracy {entry['accuracy']:.2f}{extra}")
        else:
            print(f"Knock-out: {entry}")
    probes = summary["relation_probes"]
    if probes.get("edges"):
        print(f"Edge prob touching {_fmt_ci(probes['edges']['edge_prob_touching'])} vs distant "
              f"{_fmt_ci(probes['edges']['edge_prob_distant'])}")
    if "changed_pct_relevant" in probes:
        print(f"Answers changed: relevant {_fmt_ci(probes['changed_pct_relevant'])} %, irrelevant "
              f"{_fmt_ci(probes['changed_pct_irrelevant'])} %")
        print(f"Answer-object attention (1.0 = uniform): {_fmt_ci(probes['answer_object_attention_ratio'])}")
