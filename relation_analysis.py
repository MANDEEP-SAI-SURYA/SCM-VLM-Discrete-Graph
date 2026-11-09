"""Full held-out evaluation of a trained checkpoint (same image-level split as train.py).

    python relation_analysis.py --checkpoint checkpoints/scm_vlm_full.pt --samples 30000 --tag full
    python relation_analysis.py --checkpoint checkpoints/scm_vlm_nograph.pt --samples 30000 --tag nograph

Use the same --samples / --val-fraction / --max-answers as the training run, so the
validation images are the same.  Reports (results/*_<tag>):
  factual accuracy (overall, per answer type, relational vs. non-relational),
  counterfactual accuracy per type, graph knock-out (learned / messages off / scrambled)
  with McNemar p-values, and relation probes with bootstrap 95 % CIs.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
from pathlib import Path

import torch

import train
from datasets.datasetBuilder import VQACOCO
from main import load_trained_model
from utils.evaluation import run_full_evaluation


def main(argv=None):
    parser = argparse.ArgumentParser(description="Evaluate relation use of a trained SCM-VLM")
    parser.add_argument("--checkpoint", default=train.CHECKPOINT_PATH)
    parser.add_argument("--samples", type=int, default=None,
                        help="as in training (default: read from the checkpoint)")
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--max-answers", type=int, default=train.MAX_ANSWERS)
    parser.add_argument("--coco-instances", default=train.INSTANCES_FILE)
    parser.add_argument("--tag", default=None)
    parser.add_argument("--no-irrelevant-graph-check", action="store_true")
    parser.add_argument("--seed", type=int, default=0, help="seed of the scrambled graphs")
    args = parser.parse_args(argv)

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    samples = args.samples or checkpoint.get("samples") or train.TRAIN_SAMPLES
    val_fraction = args.val_fraction or checkpoint.get("val_fraction") or train.VAL_FRACTION
    tag = args.tag if args.tag is not None else checkpoint.get("tag")

    model, idx2answer = load_trained_model(args.checkpoint)
    dataset = VQACOCO(train.IMAGE_DIR, train.QUESTION_FILE, train.ANNOTATION_FILE, max_answers=args.max_answers)
    if len(dataset.answer2idx) != len(idx2answer):
        raise RuntimeError("The answer vocabulary differs from the checkpoint's: check --max-answers.")
    train_idx, val_idx = train.split_by_image(dataset, samples, val_fraction, train.SPLIT_SEED)
    print(f"Validation questions: {len(val_idx)} (samples={samples}, val_fraction={val_fraction})")

    cache = train.FeatureCache(model, train.CACHE_DIR)
    val_cf = []
    if Path(args.coco_instances).is_file():
        _, val_cf = train.build_counterfactual_splits(
            dataset, [], val_idx, cache, train.make_steer(model), args.coco_instances, train.SPLIT_SEED)
    else:
        print(f"NOTE: {args.coco_instances} not found: counterfactual metrics are skipped.")

    run_full_evaluation(model, dataset, val_idx, cache, val_cf, tag=tag, results_dir=train.RESULTS_DIR,
                        irrelevant_graph_check=not args.no_irrelevant_graph_check, seed=args.seed)


if __name__ == "__main__":
    main()
