"""Compare finished runs (results/eval_summary_<tag>.json + eval_factual / eval_counterfactual CSVs).

    python compare_runs.py --baseline soft soft discrete soft_neurosym fw_all
    python compare_runs.py --baseline soft --runs soft soft_s1 soft_s2 soft_neurosym soft_neurosym_s1 soft_neurosym_s2

* Seeds: tags that differ only by a suffix _s<number> or _<number> (e.g. soft, soft_s1, soft_s2) are one
  configuration; the table shows mean +- std over its seeds.
* Significance: every run is compared with the baseline run of the SAME seed with an exact
  McNemar test on the same held-out questions (factual) and the same counterfactual examples,
  plus a bootstrap 95 % CI of the accuracy difference.  Same seed = same suffix.
* Symbolic fusion: the "neural only" columns score the same model without the rule-based
  answer, and "CF without rule answer" only the examples the rules could not answer.

Writes results/comparison.csv and results/comparison.md.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import statistics as st

import numpy as np
import pandas as pd

from utils.evaluation import bootstrap_diff_ci, mcnemar_exact

SEED_SUFFIX = re.compile(r"^(?P<base>.+?)(?P<seed>_s?\d+)?$")


def split_tag(tag: str) -> tuple[str, str]:
    m = SEED_SUFFIX.match(tag)
    return m.group("base"), m.group("seed") or ""


def load(results: str, tag: str) -> dict:
    summary = json.load(open(os.path.join(results, f"eval_summary_{tag}.json"), encoding="utf-8"))
    factual = pd.read_csv(os.path.join(results, f"eval_factual_{tag}.csv"))
    cf_path = os.path.join(results, f"eval_counterfactual_{tag}.csv")
    cf = pd.read_csv(cf_path) if os.path.isfile(cf_path) else pd.DataFrame()
    return {"summary": summary, "factual": factual, "cf": cf}


def metrics(run: dict) -> dict:
    s = run["summary"]
    fa = s["factual_accuracy"]
    cf = s.get("counterfactual_accuracy", {})
    g = s.get("graph_structure", {}) or {}
    row = {
        "factual": fa["overall"],
        "relational": fa.get("relational", float("nan")),
        "factual (neural only)": fa.get("overall_neural_only", float("nan")),
        "counterfactual": cf.get("overall", float("nan")),
        "CF (neural only)": cf.get("overall_neural_only", float("nan")),
        "CF without rule answer": cf.get("without_rule_answer", float("nan")),
    }
    for kind in ("existence", "counting", "replacement", "irrelevant"):
        row[f"cf {kind}"] = cf.get(kind, float("nan"))
    for key in ("dag", "root_fraction", "parents_per_node", "stability"):
        row[key] = g[key]["mean"] if key in g else float("nan")
    return row


def paired(base: dict, run: dict) -> dict:
    """McNemar + bootstrap CI of the difference vs the baseline on identical items."""
    out = {}
    f = base["factual"].merge(run["factual"], on="question_id", suffixes=("_b", "_r"))
    if len(f):
        test = mcnemar_exact(f["correct_learned_b"], f["correct_learned_r"])
        diff = bootstrap_diff_ci(f["correct_learned_r"] * 100.0, f["correct_learned_b"] * 100.0)
        out["factual diff"] = f"{diff['mean']:+.2f} [{diff['ci_low']:+.2f}, {diff['ci_high']:+.2f}]"
        out["factual p"] = test["p_value"]
    b, r = base["cf"], run["cf"]
    if len(b) and len(b) == len(r) and (b["question"].values == r["question"].values).all():
        test = mcnemar_exact(b["cf_correct"], r["cf_correct"])
        diff = bootstrap_diff_ci(r["cf_correct"] * 100.0, b["cf_correct"] * 100.0)
        out["CF diff"] = f"{diff['mean']:+.2f} [{diff['ci_low']:+.2f}, {diff['ci_high']:+.2f}]"
        out["CF p"] = test["p_value"]
    return out


def fmt(values: list[float]) -> str:
    values = [v for v in values if v == v]       # drop NaN
    if not values:
        return "-"
    if len(values) == 1:
        return f"{values[0]:.2f}"
    return f"{st.mean(values):.2f} ± {st.stdev(values):.2f}"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("runs", nargs="*", help="run tags (results/eval_summary_<tag>.json)")
    parser.add_argument("--runs", dest="runs_opt", nargs="*", default=None)
    parser.add_argument("--baseline", default=None, help="configuration to test the others against")
    parser.add_argument("--results", default="results")
    args = parser.parse_args(argv)
    tags = (args.runs_opt or []) + args.runs
    tags = [t for t in dict.fromkeys(tags)
            if os.path.isfile(os.path.join(args.results, f"eval_summary_{t}.json"))]
    if not tags:
        raise SystemExit("no finished runs found")
    runs = {t: load(args.results, t) for t in tags}

    groups: dict[str, list[str]] = {}
    for t in tags:
        groups.setdefault(split_tag(t)[0], []).append(t)
    rows = []
    for config, members in groups.items():
        per_run = [metrics(runs[t]) for t in members]
        row = {"configuration": config, "seeds": len(members)}
        for key in per_run[0]:
            row[key] = fmt([m[key] for m in per_run])
        rows.append(row)
    table = pd.DataFrame(rows).set_index("configuration")
    table = table.loc[:, (table != "-").any(axis=0)]

    tests = []
    if args.baseline:
        base_by_seed = {split_tag(t)[1]: t for t in groups.get(args.baseline, [])}
        for t in tags:
            config, seed = split_tag(t)
            if config == args.baseline or seed not in base_by_seed:
                continue
            tests.append({"run": t, "vs": base_by_seed[seed], **paired(runs[base_by_seed[seed]], runs[t])})
    tests = pd.DataFrame(tests)

    os.makedirs(args.results, exist_ok=True)
    table.to_csv(os.path.join(args.results, "comparison.csv"))
    with open(os.path.join(args.results, "comparison.md"), "w", encoding="utf-8") as file:
        file.write("## Runs (mean ± std over seeds)\n\n" + table.to_markdown() + "\n")
        if len(tests):
            file.write("\n## Paired tests vs the baseline (same seed, same held-out items)\n\n"
                       + tests.to_markdown(index=False, floatfmt=".3g") + "\n")
    pd.set_option("display.width", 200)
    print(table.T.to_string())
    if len(tests):
        print("\nPaired tests vs", args.baseline, "(difference in points, 95% bootstrap CI; exact McNemar p):")
        print(tests.to_string(index=False))
        print("\np < 0.05 and a CI that excludes 0 = a real difference on this split; otherwise it is noise.")
    print(f"\nWrote {args.results}/comparison.csv and comparison.md")
    return table, tests


if __name__ == "__main__":
    main()
