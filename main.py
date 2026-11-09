import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import os
import torch

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from configs.config import DEVICE, FEATURE_CACHE_VERSION
from models.scm_vlm import SCMVLM, build_model_from_checkpoint

CHECKPOINT_PATH = "checkpoints/scm_vlm_latest.pt"


def load_trained_model(checkpoint_path: str = CHECKPOINT_PATH, verbose: bool = True) -> tuple[SCMVLM, dict[int, str]]:
    if not os.path.isfile(checkpoint_path):
        raise FileNotFoundError(
            f"No trained checkpoint found at '{checkpoint_path}'. "
            "Run 'python train.py' first."
        )

    checkpoint = torch.load(checkpoint_path, map_location=DEVICE, weights_only=False)
    # The architecture is read from the checkpoint; old checkpoints (format_version < 3)
    # map to the architecture they were trained with.
    model = build_model_from_checkpoint(checkpoint).to(DEVICE)
    architecture = model.architecture()
    if verbose and architecture["graph_mode"] == "legacy":
        print(
            "NOTE: this checkpoint uses the old graph (hard 0.5 edge threshold). In models trained "
            "this way the relation graph usually collapsed to no edges. Retrain with "
            "'python train.py' to get the soft graph."
        )
    if verbose and not architecture["question_aware"]:
        print(
            "WARNING: this checkpoint uses the old answer head, which never sees the "
            "question, so its answers are the same for every question (mostly yes/no). "
            "Retrain with 'python train.py' to get question-dependent answers."
        )
    if verbose and checkpoint.get("feature_cache_version", 1) < FEATURE_CACHE_VERSION:
        print(
            "NOTE: this checkpoint was trained on the old DETR features, where each detection got "
            "the embedding of the wrong DETR query. It still runs (inference uses the fixed "
            "features), but retraining is recommended."
        )
    incompatible = model.load_state_dict(
        checkpoint["trainable_state_dict"], strict=False
    )
    if incompatible.unexpected_keys:
        raise RuntimeError(
            f"Checkpoint has unexpected parameters: {incompatible.unexpected_keys}"
        )
    missing = [key for key in incompatible.missing_keys
               if not key.startswith(("vision.", "text."))]
    if missing:
        raise RuntimeError(f"Checkpoint is missing trained parameters: {missing[:10]}")

    idx2answer = {
        int(index): answer for index, answer in checkpoint["idx2answer"].items()
    }
    model.eval()
    if verbose:
        print(
            f"Loaded checkpoint from epoch {checkpoint['epoch']} "
            f"(training accuracy: {checkpoint['training_accuracy']:.4f})"
        )
    return model, idx2answer


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Run SCM-VLM on any image + question")
    parser.add_argument("--image", required=True, help="path to any image")
    parser.add_argument("--question", required=True, help="question about the image")
    args = parser.parse_args()
    if not os.path.isfile(args.image):
        parser.error(f"image not found: {args.image}")

    model, idx2answer = load_trained_model()

    image_path = args.image
    question = args.question

    # =========================
    # VISION ENCODER OUTPUT
    # =========================

    with torch.inference_mode():
        vision_output = model.vision(image_path)

    print("\n=========================")
    print("DETECTED OBJECTS")
    print("=========================\n")

    labels = vision_output["labels"][0]

    for i, label in enumerate(labels):
        print(f"Node {i}: {label}")

    # =========================
    # NORMAL FORWARD PASS
    # =========================

    with torch.inference_mode():
        output = model(image_path, question)

    print("\n=========================")
    print("MODEL OUTPUT (FACTUAL)")
    print("=========================\n")

    print("Logits:\n", output["logits"])

    prediction_index = torch.argmax(output["logits"], dim=-1).item()
    print("\nPredicted answer:", idx2answer.get(prediction_index, "unknown"))

    adjacency = output["adjacency"]

    print("\nAdjacency Matrix Shape:\n", adjacency.shape)
    print("\nAdjacency Matrix:\n", adjacency)

    print("\nFor what-if questions use: python whatif_qa.py --image <path> --whatif \"What if ...\"")

    # =========================
    # COUNTERFACTUAL REASONING  (superseded by whatif_qa.py)
    # =========================

    # print("\n=========================")
    # print("COUNTERFACTUAL ANALYSIS")
    # print("=========================\n")

    # objects = output["objects"]
    # adjacency = output["adjacency"]

    # # -------------------------
    # # STEP 1: Choose node
    # # -------------------------
    # # Example: assume node 1 = bicycle (you printed labels above)
    # node_index = torch.tensor([1]).to(DEVICE)

    # # -------------------------
    # # STEP 2: Create intervention
    # # -------------------------
    # # Option A: remove object (recommended for paper)
    # replacement = torch.zeros(
    #     1,
    #     objects.shape[-1],
    #     device=DEVICE
    # )

    # # -------------------------
    # # STEP 3: Run counterfactual
    # # -------------------------
    # cf_output = model.counterfactual(
    #     objects,
    #     adjacency,
    #     node_index,
    #     replacement
    # )

    # # -------------------------
    # # STEP 4: Compare results
    # # -------------------------
    # print("Factual logits:\n", output["logits"])
    # print("\nCounterfactual logits:\n", cf_output["logits"])

    # print("\nLogit shift:\n",
    #       cf_output["logits"] - output["logits"])


if __name__ == "__main__":
    main()
