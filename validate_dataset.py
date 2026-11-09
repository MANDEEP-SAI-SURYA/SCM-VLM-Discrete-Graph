"""Check that downloaded VQA files and COCO images belong to the same split."""

from __future__ import annotations

import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent
IMAGE_DIR = PROJECT_ROOT / "coco" / "train2014" / "train2014"
QUESTION_FILE = PROJECT_ROOT / "coco" / "vqa" / "v2_OpenEnded_mscoco_train2014_questions.json"
ANNOTATION_FILE = PROJECT_ROOT / "coco" / "vqa" / "v2_mscoco_train2014_annotations.json"


def main() -> None:
    for path in (IMAGE_DIR, QUESTION_FILE, ANNOTATION_FILE):
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")

    with QUESTION_FILE.open(encoding="utf-8") as file:
        questions = json.load(file)["questions"]
    with ANNOTATION_FILE.open(encoding="utf-8") as file:
        annotations = json.load(file)["annotations"]

    image_names = {path.name for path in IMAGE_DIR.glob("*.jpg")}
    image_split = IMAGE_DIR.name
    matching = sum(
        f"COCO_{image_split}_{question['image_id']:012d}.jpg" in image_names
        for question in questions
    )

    print("VQA-COCO DATASET CHECK")
    print(f"Images found       : {len(image_names):,} ({image_split})")
    print(f"Questions found    : {len(questions):,} (train2014)")
    print(f"Annotations found  : {len(annotations):,} (train2014)")
    print(f"Valid image-QA pairs: {matching:,}")

    if matching == 0:
        print("\nRESULT: Training cannot start: the files are from different COCO splits.")
        print("Expected image names such as COCO_train2014_000000000009.jpg.")
    else:
        print("\nRESULT: Dataset is aligned and ready for training.")


if __name__ == "__main__":
    main()
