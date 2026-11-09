import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import torch

import argparse

from main import load_trained_model
from utils.visualization import visualize_graph


def main() -> None:

    parser = argparse.ArgumentParser(description="Causal-graph visualisation for any image + question")
    parser.add_argument("--image", required=True, help="path to any image")
    parser.add_argument("--question", required=True)
    parser.add_argument("--out", default="images/causal_graph.png")
    args = parser.parse_args()
    if not os.path.isfile(args.image):
        parser.error(f"image not found: {args.image}")

    # uses the trained checkpoint (the old version used an untrained model)
    model, idx2answer = load_trained_model()

    image_path = args.image

    question = args.question

    with torch.no_grad():

        output = model(image_path, question)

    prediction = torch.argmax(
        output["logits"],
        dim=-1
    )

    print("\nPrediction:")
    print(idx2answer.get(prediction.item(), prediction.item()))

    # =========================
    # SAVE GRAPH IMAGE
    # =========================

    visualize_graph(
        output["adjacency"],
        save_path=args.out
    )

    print(f"\nSaved graph visualization to {args.out}")

    # =========================
    # PRINT TOP CAUSAL RELATIONS
    # =========================

    adj = output["adjacency"][0]

    labels = output["labels"][0]

    print("\n=========================")
    print("TOP CAUSAL RELATIONS")
    print("=========================\n")

    for i in range(len(labels)):

        row = adj[i]

        top_indices = torch.argsort(
            row,
            descending=True
        )[:3]

        printed = 0

        for j in top_indices:

            j = j.item()

            if i == j:
                continue

            print(
                # A[i, j] > 0 means j -> i (j is a parent of i)
                f"{labels[j]} ---> "
                f"{labels[i]} : "
                f"{adj[i][j].item():.3f}"
            )

            printed += 1

            if printed >= 2:
                break


if __name__ == "__main__":
    main()
