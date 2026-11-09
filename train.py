"""Train SCM-VLM on VQA v2 / COCO.

    python train.py                                      # defaults below (full model)
    python train.py --samples 30000 --epochs 10 --tag full
    python train.py --samples 30000 --epochs 10 --graph-mode none --tag nograph
    python train.py --cf-mode invariance_old --tag old_cf        # reproduce the previous loss

Counterfactual supervision (--cf-mode supervised, default) needs the COCO instance
annotations (coco/annotations/instances_train2014.json).  Examples with KNOWN answers are
generated from them (datasets/counterfactual.py) and mixed with the VQA questions:

    L = L_task + lambda_cf * CE(Y_do, Y_cf) + graph regularisers

--cf-mode invariance_old reproduces the earlier loss KL(cf || factual) under small random
node noise, which trained the model to IGNORE interventions.  --cf-mode none: no
counterfactual term.

History of fixes (see README): question-aware answer head; 3129-answer vocabulary; cached
frozen DETR/BERT features with an image-level train/validation split; NOTEARS acyclicity;
soft top-k relation graph with box geometry and contact supervision; DETR features now come
from the query that produced each detection (FEATURE_CACHE_VERSION 2); node identity
(class + box), relation-aware messages, counterfactual supervision.
"""

import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import argparse
import hashlib
import json
import random
from collections import defaultdict
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
import torch.nn.functional as F

torch.backends.cudnn.benchmark = True

from configs.coco_labels import label_to_id
from configs.config import DEVICE, FEATURE_CACHE_VERSION
from datasets.counterfactual import CF_TYPES, build_counterfactual_examples
from datasets.datasetBuilder import VQACOCO
from losses.losses import acyclicity_loss
from models.scm_graph import contact_targets
from models.scm_vlm import SCMVLM


# =====================================================
# SETTINGS (defaults; override on the command line)
# =====================================================

RESULTS_DIR = "results"
CHECKPOINT_DIR = "checkpoints"
CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR, "scm_vlm_latest.pt")
CACHE_DIR = Path("cache")

IMAGE_DIR = "coco/train2014/train2014"
QUESTION_FILE = "coco/vqa/v2_OpenEnded_mscoco_train2014_questions.json"
ANNOTATION_FILE = "coco/vqa/v2_mscoco_train2014_annotations.json"
INSTANCES_FILE = "coco/annotations/instances_train2014.json"

EPOCHS = 10
BATCH_SIZE = 16            # gradients are accumulated over the batch
TRAIN_SAMPLES = 20000      # questions used (train + validation)
VAL_FRACTION = 0.1         # held-out fraction, split by image
MAX_ANSWERS = 3129
LEARNING_RATE = 1e-4
SEED = 42                  # model initialisation / sampling (--seed)
SPLIT_SEED = 42            # train / validation split: fixed so every run uses the same split

TOP_K = 3
RANDOM_NODE_PROB = 0.3

LAMBDA_CF = 1.0           # supervised counterfactual CE  (--cf-mode supervised)
LAMBDA_CF_OLD = 0.1       # KL(cf || factual) under noise  (--cf-mode invariance_old, previous default)
LAMBDA_SPARSE = 0.01
LAMBDA_ACYCLIC = 0.05
LAMBDA_ENTROPY = 0.05
LAMBDA_EDGE = 0.5         # weak relation supervision: objects whose boxes touch should be linked
GRAPH_TOP_K = 3


# =====================================================
# FEATURE CACHE (frozen DETR + BERT)
# =====================================================

def _key(text: str) -> str:
    return hashlib.md5(text.encode("utf-8")).hexdigest()


class FeatureCache:
    """Frozen DETR detections per image and BERT embeddings per question.

    Files carry FEATURE_CACHE_VERSION; files from an older version (e.g. the misaligned
    DETR embeddings of version 1, or files without boxes / class ids) are rebuilt."""

    def __init__(self, model: SCMVLM, root: Path) -> None:
        self.model = model
        self.image_dir = root / "detr"
        self.image_dir.mkdir(parents=True, exist_ok=True)
        self.questions: dict[str, torch.Tensor] = {}
        self.memory: dict[str, dict] = {}

    @torch.no_grad()
    def image(self, image_path: str) -> dict:
        """{"embeddings" [N, 256], "boxes" [N, 4] (x0, y0, x1, y1 in 0-1), "labels" [str],
        "label_ids" [N] (DETR / COCO ids)}"""
        image_path = str(image_path)
        if image_path in self.memory:
            return self.memory[image_path]
        path = self.image_dir / f"{_key(os.path.abspath(image_path))}.pt"
        item = None
        if path.is_file():
            item = torch.load(path, map_location="cpu", weights_only=False)
            if item.get("version") != FEATURE_CACHE_VERSION:
                item = None
        if item is None:
            output = self.model.vision(image_path)
            embeddings = output["embeddings"][0].detach().cpu()
            if output.get("boxes_norm"):
                boxes = output["boxes_norm"][0].detach().cpu().float()
            elif output.get("boxes"):
                with Image.open(image_path) as image:
                    width, height = image.size
                boxes = output["boxes"][0].detach().cpu().float() / torch.tensor([width, height, width, height])
            else:
                boxes = torch.zeros(len(embeddings), 4)
            labels = list(output["labels"][0])
            label_ids = (output["label_ids"][0].detach().cpu().long() if output.get("label_ids")
                         else torch.tensor([label_to_id(label) for label in labels], dtype=torch.long))
            item = {"version": FEATURE_CACHE_VERSION, "embeddings": embeddings[:len(labels)],
                    "boxes": boxes, "labels": labels, "label_ids": label_ids}
            torch.save(item, path)
        if len(self.memory) < 50000:
            self.memory[image_path] = item
        return item

    @torch.no_grad()
    def question(self, question: str) -> torch.Tensor:
        if question not in self.questions:
            self.questions[question] = self.model.text(question)[0].detach().cpu()
        return self.questions[question]


def make_steer(model: SCMVLM):
    """ClassSteerer on DETR's class head (x' for replace interventions), or None when the
    vision encoder has no class head (tests with stubbed DETR may pass their own)."""
    from models.do_intervention import ClassSteerer

    detr = getattr(getattr(model, "vision", None), "model", None)
    if detr is None or not hasattr(detr, "class_labels_classifier"):
        return None
    steerer = ClassSteerer(detr.class_labels_classifier, detr.config.label2id)

    def steer(embedding: torch.Tensor, label: str):
        device = steerer.classifier.weight.device
        x_prime, _, converged = steerer.steer(embedding.to(device), label)
        return x_prime.detach().cpu(), converged

    return steer


# =====================================================
# HELPERS
# =====================================================

def split_by_image(dataset: VQACOCO, num_samples: int, val_fraction: float, seed: int = SPLIT_SEED):
    """Pick up to ``num_samples`` questions and split them so that no image
    appears in both train and validation."""
    by_image = defaultdict(list)
    for index, record in enumerate(dataset.records):
        by_image[str(record["image_path"])].append(index)
    images = sorted(by_image)
    random.Random(seed).shuffle(images)

    chosen = []
    total = 0
    for image in images:
        chosen.append(image)
        total += len(by_image[image])
        if total >= num_samples:
            break
    num_val = max(1, int(round(len(chosen) * val_fraction)))
    val_images, train_images = chosen[:num_val], chosen[num_val:]
    train = [i for image in train_images for i in by_image[image]]
    val = [i for image in val_images for i in by_image[image]]
    return train, val


def output_paths(tag: str | None) -> dict:
    suffix = f"_{tag}" if tag else ""
    checkpoint = CHECKPOINT_PATH if not tag else os.path.join(CHECKPOINT_DIR, f"scm_vlm_{tag}.pt")
    return {
        "checkpoint": checkpoint,
        "history": os.path.join(RESULTS_DIR, f"training_history{suffix}.csv"),
        "final": os.path.join(RESULTS_DIR, f"final_metrics{suffix}.csv"),
        "suffix": suffix,
    }


def save_checkpoint(model, dataset, epoch, train_accuracy, val_accuracy, path=CHECKPOINT_PATH, extra=None):
    """Save learned layers + answer vocabulary. DETR and BERT are frozen
    pretrained backbones, so they are reloaded from Hugging Face instead."""
    trainable_names = {name for name, parameter in model.named_parameters() if parameter.requires_grad}
    trainable_state = {
        name: tensor.detach().cpu()
        for name, tensor in model.state_dict().items()
        if name in trainable_names
    }
    architecture = model.architecture()
    torch.save({
        "format_version": 3,
        "architecture": architecture,
        # kept for code that reads the version-2 fields directly
        "question_aware": architecture["question_aware"],
        "graph_mode": architecture["graph_mode"],
        "graph_top_k": architecture["graph_top_k"],
        "graph_geometry": architecture["graph_geometry"],
        "feature_cache_version": FEATURE_CACHE_VERSION,
        "epoch": epoch,
        "training_accuracy": train_accuracy,
        "validation_accuracy": val_accuracy,
        "num_classes": len(dataset.answer2idx),
        "idx2answer": dataset.idx2answer,
        "trainable_state_dict": trainable_state,
        **(extra or {}),
    }, path)
    print(f"Saved checkpoint: {path}")


def edge_supervision_loss(edge_probs: torch.Tensor, boxes: torch.Tensor) -> tuple[torch.Tensor, dict]:
    """BCE between the (direction-free) edge probability max(p_ij, p_ji) and box contact.
    The direction is left to the model (and the acyclicity term)."""
    n = edge_probs.shape[0]
    if n < 2:
        zero = edge_probs.new_tensor(0.0)
        return zero, {"p_contact": None, "p_far": None}
    target = contact_targets(boxes.to(edge_probs.device))
    sym = torch.maximum(edge_probs, edge_probs.T).clamp(1e-6, 1 - 1e-6)
    off = ~torch.eye(n, dtype=torch.bool, device=edge_probs.device)
    positives = target[off].sum()
    weight = torch.where(target[off] > 0, (off.sum() - positives).clamp(min=1) / positives.clamp(min=1),
                         torch.ones_like(target[off]))   # balance contact / non-contact pairs
    loss = F.binary_cross_entropy(sym[off], target[off], weight=weight)
    contact = target[off] > 0
    stats = {
        "p_contact": sym[off][contact].mean().detach() if contact.any() else None,
        "p_far": sym[off][~contact].mean().detach() if (~contact).any() else None,
    }
    return loss, stats


def forward_detections(model, detections: dict, q_emb: torch.Tensor):
    return model.forward_features(
        detections["embeddings"].unsqueeze(0).to(DEVICE),
        q_emb.unsqueeze(0).to(DEVICE),
        detections["boxes"].unsqueeze(0).to(DEVICE),
        detections["label_ids"].unsqueeze(0).to(DEVICE),
    )


def graph_regularisers(model, out, boxes, settings) -> tuple[torch.Tensor, dict]:
    """Sparsity, acyclicity, entropy and contact supervision on the learned graph
    (all zero for --graph-mode none)."""
    zero = out["logits"].new_tensor(0.0)
    if model.graph is None:
        return zero, {"acyclic": zero, "edge": zero, "graph_mass": zero, "edge_prob": zero,
                      "p_contact": None, "p_far": None}
    A = torch.clamp(out["adjacency"].float(), 0.0, 1.0)[0]
    loss_sparse = A.mean()
    loss_acyclic = acyclicity_loss(A)
    edge_prob = A / (A.sum(dim=-1, keepdim=True) + 1e-6)
    loss_entropy = -torch.mean(edge_prob * torch.log(edge_prob + 1e-6))
    if settings["lambda_edge"] > 0:
        loss_edge, edge_stats = edge_supervision_loss(out["edge_probs"][0], boxes)
    else:
        loss_edge, edge_stats = zero, {"p_contact": None, "p_far": None}
    loss = (settings["lambda_edge"] * loss_edge + LAMBDA_SPARSE * loss_sparse
            + LAMBDA_ACYCLIC * loss_acyclic + LAMBDA_ENTROPY * loss_entropy)
    off = ~torch.eye(A.shape[0], dtype=torch.bool, device=A.device)
    return loss, {
        "acyclic": loss_acyclic.detach(), "edge": loss_edge.detach(),
        "graph_mass": A.sum(dim=-1).mean().detach(),
        "edge_prob": out["edge_probs"][0][off].mean().detach() if A.shape[0] > 1 else zero,
        **edge_stats,
    }


def factual_step(model, detections, q_emb, label, settings):
    """VQA question: task CE + graph regularisers (+ the old invariance KL for
    --cf-mode invariance_old)."""
    out = forward_detections(model, detections, q_emb)
    logits = torch.clamp(out["logits"], -30, 30)
    loss_task = F.cross_entropy(logits, label.view(1).to(logits.device))
    loss_reg, parts = graph_regularisers(model, out, detections["boxes"], settings)
    loss = loss_task + loss_reg
    loss_cf = logits.new_tensor(0.0)

    if settings["cf_mode"] == "invariance_old" and model.graph is not None:
        # previous behaviour: KL(cf || factual) under small random node noise
        A = torch.clamp(out["adjacency"].float(), 0.0, 1.0)
        objects = out["objects"]
        node_scores = A[0].sum(dim=0)
        k = min(TOP_K, node_scores.shape[0])
        _, topk_idx = torch.topk(node_scores, k=k)
        if random.random() < RANDOM_NODE_PROB:
            node_idx = torch.randint(0, objects.shape[1], (1,), device=objects.device)
        else:
            node_idx = topk_idx[torch.randint(0, k, (1,))].view(1)
        strength = random.random() * 0.2
        replacement = objects[:, node_idx.item()] + torch.randn_like(objects[:, 0]) * strength
        cf_out = model.counterfactual(objects, A, node_idx, replacement, out["question_embedding"],
                                      out["boxes_norm"], out["relation_probs"])
        cf_logits = torch.clamp(cf_out["logits"], -30, 30)
        loss_cf = F.kl_div(F.log_softmax(cf_logits, dim=-1), F.softmax(logits.detach(), dim=-1),
                           reduction="batchmean") * (1.0 + strength)
        loss = loss + settings["lambda_cf"] * loss_cf

    parts.update({"task": loss_task.detach(), "cf": loss_cf.detach()})
    return loss, logits, parts


def irrelevant_is_related(out, detections, example) -> bool:
    """True when the intervened node is a parent or child of an object mentioned in the
    question in the CURRENT learned graph (then the example is not 'irrelevant')."""
    if example["type"] != "irrelevant" or out["adjacency"] is None:
        return False
    A = out["adjacency"][0]
    mentioned = [i for i, label in enumerate(detections["labels"]) if label in set(example["mentioned"])]
    return any(bool(A[m, k] > 1e-6) or bool(A[k, m] > 1e-6) for k in example["nodes"] for m in mentioned)


def counterfactual_step(model, detections, q_emb, example, answer2idx, settings):
    """Counterfactual example: factual CE on the same question (when its answer is known)
    + lambda_cf * CE(Y_do, Y_cf).  Returns (loss, cf_correct or None, parts)."""
    out = forward_detections(model, detections, q_emb)
    logits = torch.clamp(out["logits"], -30, 30)
    loss_reg, parts = graph_regularisers(model, out, detections["boxes"], settings)
    loss = loss_reg
    if example["factual_answer"] in answer2idx:
        target = torch.tensor([answer2idx[example["factual_answer"]]], device=logits.device)
        loss = loss + F.cross_entropy(logits, target)

    if settings["irrelevant_graph_check"] and model.graph is not None and \
            irrelevant_is_related(out, detections, example):
        parts.update({"task": loss.detach(), "cf": logits.new_tensor(0.0), "skipped": True})
        return loss, None, parts

    new_raw = example["x_prime"].to(DEVICE) if example["kind"] == "replace" else None
    new_ids = [example["new_label_id"]] * len(example["nodes"]) if example["kind"] == "replace" else None
    cf = model.intervene(out, example["kind"], example["nodes"], new_raw=new_raw, new_class_ids=new_ids)
    cf_logits = torch.clamp(cf["logits"], -30, 30)
    cf_target = torch.tensor([answer2idx[example["cf_answer"]]], device=cf_logits.device)
    loss_cf = F.cross_entropy(cf_logits, cf_target)
    loss = loss + settings["lambda_cf"] * loss_cf
    correct = int(cf_logits.argmax(-1).item() == cf_target.item())
    parts.update({"task": loss.detach(), "cf": loss_cf.detach(), "skipped": False})
    return loss, correct, parts


@torch.no_grad()
def evaluate(model, dataset, indices, cache, cf_examples=None, answer2idx=None, settings=None):
    """Per-epoch validation: factual accuracy, counterfactual accuracy per type on the
    held-out generated examples, and edge statistics.  (The full analysis with knock-outs
    and probes runs once at the end: utils/evaluation.py.)"""
    model.eval()
    factual = total = 0
    p_contact, p_far = [], []
    for index in indices:
        image_path, question, label = dataset[index]
        try:
            detections = cache.image(image_path)
            if len(detections["labels"]) == 0:
                continue
            out = forward_detections(model, detections, cache.question(question))
        except Exception as error:  # unreadable image etc.
            print(f"EVAL SKIP {image_path}: {error}")
            continue
        if model.graph is not None:
            _, edge_stats = edge_supervision_loss(out["edge_probs"][0], detections["boxes"])
            if edge_stats["p_contact"] is not None:
                p_contact.append(float(edge_stats["p_contact"]))
            if edge_stats["p_far"] is not None:
                p_far.append(float(edge_stats["p_far"]))
        factual += int(out["logits"].argmax(-1).item() == label.item())
        total += 1

    cf_correct = defaultdict(list)
    for example in cf_examples or []:
        detections = cache.image(example["image_path"])
        out = forward_detections(model, detections, cache.question(example["question"]))
        # every model is scored on the same held-out examples (the learned-graph filter for
        # 'irrelevant' examples only applies to training; see utils/evaluation.py)
        new_raw = example["x_prime"].to(DEVICE) if example["kind"] == "replace" else None
        new_ids = [example["new_label_id"]] * len(example["nodes"]) if example["kind"] == "replace" else None
        cf = model.intervene(out, example["kind"], example["nodes"], new_raw=new_raw, new_class_ids=new_ids)
        cf_correct[example["type"]].append(int(cf["logits"].argmax(-1).item() == answer2idx[example["cf_answer"]]))
    model.train()
    result = {
        "val_factual_accuracy": 100.0 * factual / max(total, 1),
        "val_edge_prob_contact": sum(p_contact) / max(len(p_contact), 1),
        "val_edge_prob_far": sum(p_far) / max(len(p_far), 1),
    }
    all_cf = [c for values in cf_correct.values() for c in values]
    result["val_cf_accuracy"] = 100.0 * sum(all_cf) / len(all_cf) if all_cf else float("nan")
    for kind in CF_TYPES:
        values = cf_correct.get(kind, [])
        result[f"val_cf_{kind}"] = 100.0 * sum(values) / len(values) if values else float("nan")
    return result


def build_counterfactual_splits(dataset, train_idx, val_idx, cache, steer, instances_path, seed, caps=None):
    """Counterfactual examples for the training and the validation images (pass an empty
    train_idx to build only the validation examples)."""
    vocab = [dataset.idx2answer[i] for i in range(len(dataset.idx2answer))]
    common = dict(detections_fn=cache.image, instances_path=instances_path, vocab=vocab, steer=steer,
                  cache_dir=CACHE_DIR / "counterfactual", seed=seed, caps=caps,
                  feature_cache_version=FEATURE_CACHE_VERSION)
    train_cf = build_counterfactual_examples([dataset.records[i] for i in train_idx], split_name="train", **common)
    val_cf = build_counterfactual_examples([dataset.records[i] for i in val_idx], split_name="val", **common)
    return train_cf, val_cf


def build_model(args, num_classes) -> SCMVLM:
    return SCMVLM(
        num_classes=num_classes, question_aware=True,
        graph_mode=args.graph_mode, graph_top_k=GRAPH_TOP_K,
        graph_geometry=args.graph_mode == "soft",
        node_identity=not args.no_node_identity,
        relation_messages=not args.no_relation_messages,
        relation_types=args.relation_types,
        residual_last_block=not args.no_residual_last,
    ).to(DEVICE)


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="Train SCM-VLM")
    parser.add_argument("--samples", type=int, default=TRAIN_SAMPLES)
    parser.add_argument("--epochs", type=int, default=EPOCHS)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=LEARNING_RATE)
    parser.add_argument("--max-answers", type=int, default=MAX_ANSWERS)
    parser.add_argument("--val-fraction", type=float, default=VAL_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED,
                        help="model initialisation / sampling seed (the train/val split is fixed)")
    parser.add_argument("--tag", default=None,
                        help="name of the run: checkpoints/scm_vlm_<tag>.pt and results/*_<tag>.*")
    # graph / architecture
    parser.add_argument("--graph-mode", choices=["soft", "legacy", "none"], default="soft",
                        help="soft (default): learned top-k graph. legacy: the original hard-threshold "
                             "graph. none: ablation without graph and messages")
    parser.add_argument("--relation-types", type=int, default=1,
                        help="R > 1: learned relation types, one message transform per type")
    parser.add_argument("--no-node-identity", action="store_true",
                        help="ablation: node input = DETR embedding only (no class embedding / box)")
    parser.add_argument("--no-relation-messages", action="store_true",
                        help="ablation: messages carry only the parent's features (no pair geometry)")
    parser.add_argument("--no-residual-last", action="store_true",
                        help="remove the residual bypass of the last message step")
    parser.add_argument("--no-edge-supervision", action="store_true",
                        help="ablation: disable the box-contact relation supervision")
    # counterfactual supervision
    parser.add_argument("--cf-mode", choices=["none", "invariance_old", "supervised"], default="supervised")
    parser.add_argument("--lambda-cf", type=float, default=None,
                        help=f"weight of the counterfactual loss (default {LAMBDA_CF} for supervised, "
                             f"{LAMBDA_CF_OLD} for invariance_old)")
    parser.add_argument("--cf-ratio", type=float, default=1.0,
                        help="counterfactual examples per factual question in each epoch (1.0 = 1:1)")
    parser.add_argument("--coco-instances", default=INSTANCES_FILE)
    parser.add_argument("--no-irrelevant-graph-check", action="store_true",
                        help="do not skip 'irrelevant' examples whose node is a parent/child of a "
                             "mentioned object in the current learned graph")
    parser.add_argument("--no-cf", action="store_true", help="old flag, same as --cf-mode none")
    parser.add_argument("--skip-final-eval", action="store_true",
                        help="skip the full evaluation (knock-outs, probes) after training")
    args = parser.parse_args(argv)
    if args.no_cf:
        args.cf_mode = "none"
    if args.lambda_cf is None:
        args.lambda_cf = LAMBDA_CF_OLD if args.cf_mode == "invariance_old" else LAMBDA_CF
    return args


def settings_from_args(args) -> dict:
    return {
        "cf_mode": args.cf_mode,
        "lambda_cf": args.lambda_cf,
        "lambda_edge": 0.0 if (args.no_edge_supervision or args.graph_mode != "soft") else LAMBDA_EDGE,
        "irrelevant_graph_check": not args.no_irrelevant_graph_check,
    }


# =====================================================
# MAIN
# =====================================================

def main(argv=None) -> dict:
    args = parse_args(argv)
    settings = settings_from_args(args)
    paths = output_paths(args.tag)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(RESULTS_DIR, exist_ok=True)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    print(f"\nUSING DEVICE: {DEVICE}")
    print(f"Run: tag={args.tag or '-'} graph={args.graph_mode} cf_mode={args.cf_mode} "
          f"lambda_cf={args.lambda_cf} relation_types={args.relation_types} seed={args.seed}")

    dataset = VQACOCO(IMAGE_DIR, QUESTION_FILE, ANNOTATION_FILE, max_answers=args.max_answers)
    train_idx, val_idx = split_by_image(dataset, args.samples, args.val_fraction, SPLIT_SEED)
    print(f"Answer vocabulary: {len(dataset.answer2idx)} | train questions: {len(train_idx)} "
          f"| validation questions: {len(val_idx)}")

    model = build_model(args, len(dataset.answer2idx))
    cache = FeatureCache(model, CACHE_DIR)

    # Validation counterfactual examples are built for EVERY cf-mode (and graph mode), so all
    # runs are scored on the same held-out examples; training examples only for "supervised".
    train_cf, val_cf = [], []
    if Path(args.coco_instances).is_file():
        train_cf, val_cf = build_counterfactual_splits(
            dataset, train_idx if args.cf_mode == "supervised" else [], val_idx, cache,
            make_steer(model), args.coco_instances, SPLIT_SEED)
    elif args.cf_mode == "supervised":
        raise FileNotFoundError(
            f"--cf-mode supervised needs the COCO instance annotations at {args.coco_instances} "
            "(annotations_trainval2014.zip from cocodataset.org), or use --cf-mode none.")
    else:
        print(f"NOTE: {args.coco_instances} not found: no counterfactual evaluation examples.")
    cf_counts = {t: sum(e["type"] == t for e in train_cf) for t in CF_TYPES}
    print(f"Counterfactual training examples: {len(train_cf)} {cf_counts} | validation: {len(val_cf)}")

    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(args.epochs, 1))
    answer2idx = dataset.answer2idx

    history = []
    best_val = -1.0
    model.train()

    print("\n========================")
    print("TRAINING STARTED")
    print("========================")

    for epoch in range(args.epochs):
        items = [("f", i) for i in train_idx]
        if train_cf:
            n_cf = int(round(args.cf_ratio * len(train_idx)))
            pool = list(range(len(train_cf)))
            random.shuffle(pool)
            picks = [pool[i % len(pool)] for i in range(n_cf)]
            items += [("c", j) for j in picks]
        random.shuffle(items)

        total_loss, correct, seen = 0.0, 0, 0
        cf_seen, cf_correct, cf_skipped = 0, 0, 0
        graph_mass, edge_prob = 0.0, 0.0
        optimizer.zero_grad()
        in_batch = 0

        for step, (kind, index) in enumerate(items, start=1):
            try:
                if kind == "f":
                    image_path, question, label = dataset[index]
                    detections = cache.image(image_path)
                    if len(detections["labels"]) == 0:
                        continue
                    loss, logits, parts = factual_step(model, detections, cache.question(question), label, settings)
                else:
                    example = train_cf[index]
                    detections = cache.image(example["image_path"])
                    loss, cf_ok, parts = counterfactual_step(
                        model, detections, cache.question(example["question"]), example, answer2idx, settings)
            except Exception as error:
                print(f"TRAIN SKIP ({kind}) {error}")
                continue
            if torch.isnan(loss):
                continue

            (loss / args.batch_size).backward()
            in_batch += 1
            total_loss += loss.item()
            graph_mass += float(parts["graph_mass"])
            edge_prob += float(parts["edge_prob"])
            if kind == "f":
                correct += int(logits.argmax(-1).item() == label.item())
                seen += 1
            elif cf_ok is None:
                cf_skipped += 1
            else:
                cf_correct += cf_ok
                cf_seen += 1

            if in_batch == args.batch_size or step == len(items):
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
                optimizer.zero_grad()
                in_batch = 0

            if step % 1000 == 0:
                print(f"[TRAIN] epoch={epoch + 1} step={step}/{len(items)} loss={loss.item():.4f} "
                      f"running_acc={correct / max(seen, 1):.4f} "
                      f"cf_acc={cf_correct / max(cf_seen, 1):.4f} graph_mass={graph_mass / step:.3f}")

        scheduler.step()
        train_acc = correct / max(seen, 1)
        val = evaluate(model, dataset, val_idx, cache, val_cf, answer2idx, settings)
        steps = max(seen + cf_seen + cf_skipped, 1)
        row = {"epoch": epoch + 1, "loss": total_loss, "train_accuracy": train_acc,
               "train_cf_accuracy": cf_correct / max(cf_seen, 1) if cf_seen else float("nan"),
               "train_cf_skipped_related": cf_skipped,
               "graph_mass": graph_mass / steps, "mean_edge_prob": edge_prob / steps, **val}
        history.append(row)

        print("\n========================")
        print(f"EPOCH {epoch + 1}")
        print("========================")
        print(f"Loss                     : {total_loss:.4f}")
        print(f"Train accuracy           : {train_acc:.4f}")
        print(f"Val factual accuracy     : {val['val_factual_accuracy']:.2f}%")
        print("Val counterfactual acc.  : " + (f"{val['val_cf_accuracy']:.2f}%  (" + ", ".join(
            f"{t} {val[f'val_cf_{t}']:.1f}" for t in CF_TYPES) + ")" if val_cf else "n/a (no examples)"))
        if model.graph is not None:
            print(f"Graph: parent weight/obj : {row['graph_mass']:.3f}   mean edge prob: {row['mean_edge_prob']:.3f}")
            print(f"Relations (val)          : edge prob for touching objects "
                  f"{val['val_edge_prob_contact']:.3f} vs distant {val['val_edge_prob_far']:.3f}")
            if row["graph_mass"] < 0.05:
                print("WARNING: the relation graph is collapsing (almost no edge weight left).")

        pd.DataFrame(history).to_csv(paths["history"], index=False)
        if val["val_factual_accuracy"] > best_val:
            best_val = val["val_factual_accuracy"]
            save_checkpoint(model, dataset, epoch + 1, train_acc, val["val_factual_accuracy"], paths["checkpoint"],
                            extra={"cf_mode": args.cf_mode, "lambda_cf": args.lambda_cf, "seed": args.seed,
                                   "tag": args.tag, "samples": args.samples, "val_fraction": args.val_fraction})

    best = max(history, key=lambda r: r["val_factual_accuracy"]) if history else {}
    results = pd.DataFrame([{
        "Factual Accuracy": best.get("val_factual_accuracy", 0.0),
        "Counterfactual Accuracy": best.get("val_cf_accuracy", float("nan")),
        **{f"CF {t}": best.get(f"val_cf_{t}", float("nan")) for t in CF_TYPES},
        "Best Epoch": best.get("epoch", 0),
        "Validation Questions": len(val_idx),
        "Validation CF Examples": len(val_cf),
        "Graph": args.graph_mode, "CF mode": args.cf_mode, "Seed": args.seed,
    }])
    results.to_csv(paths["final"], index=False)

    print("\n========================")
    print("FINAL RESULTS (held-out validation, best epoch)")
    print("========================")
    print(results.to_string(index=False))

    summary = {"history": history, "final": results.to_dict("records")[0]}
    if not args.skip_final_eval and history:
        from utils.evaluation import run_full_evaluation

        # evaluate the best epoch (the checkpoint), not the last one
        best_state = torch.load(paths["checkpoint"], map_location=DEVICE, weights_only=False)
        model.load_state_dict(best_state["trainable_state_dict"], strict=False)
        model.eval()
        summary["evaluation"] = run_full_evaluation(
            model, dataset, val_idx, cache, val_cf, tag=args.tag, results_dir=RESULTS_DIR,
            irrelevant_graph_check=settings["irrelevant_graph_check"])
    with open(os.path.join(RESULTS_DIR, f"run_config{paths['suffix']}.json"), "w", encoding="utf-8") as file:
        json.dump(vars(args), file, indent=2)
    return summary


if __name__ == "__main__":
    main()
