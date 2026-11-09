"""PyTorch dataset for VQA v2 questions paired with COCO images."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import torch
from torch.utils.data import Dataset


class VQACOCO(Dataset):
    """Load VQA v2 Open-Ended questions, annotations, and COCO images.

    Each item is returned as ``(image_path, question, answer_index)``.  Images
    remain paths because ``VisionEncoder`` loads and preprocesses them with the
    DETR image processor.
    """

    def __init__(
        self,
        image_dir: str | Path,
        question_file: str | Path,
        annotation_file: str | Path,
        max_answers: int | None = None,
    ) -> None:
        """``max_answers``: keep only the N most frequent answers (3129 is the
        standard VQA v2 vocabulary) and drop questions whose answer is outside
        it.  ``None`` keeps every answer (the original behaviour)."""
        self.image_dir = Path(image_dir)
        question_path = Path(question_file)
        annotation_path = Path(annotation_file)

        self._require_path(self.image_dir, "COCO image directory")
        self._require_path(question_path, "VQA question file")
        self._require_path(annotation_path, "VQA annotation file")

        with question_path.open("r", encoding="utf-8") as file:
            questions = json.load(file)["questions"]
        with annotation_path.open("r", encoding="utf-8") as file:
            annotations = json.load(file)["annotations"]

        annotations_by_question = {
            annotation["question_id"]: annotation for annotation in annotations
        }

        records: list[dict[str, object]] = []
        answer_counts: Counter[str] = Counter()
        for question in questions:
            annotation = annotations_by_question.get(question["question_id"])
            if annotation is None:
                continue

            answer = self.normalize_answer(annotation["multiple_choice_answer"])
            image_path = self._image_path(question["image_id"])
            if image_path is None:
                continue

            records.append(
                {
                    "image_path": image_path,
                    "question": question["question"],
                    "answer": answer,
                    "image_id": int(question["image_id"]),
                    "question_id": int(question["question_id"]),
                    # VQA v2 answer type: "yes/no", "number" or "other"
                    "answer_type": annotation.get("answer_type", "other"),
                }
            )
            answer_counts[answer] += 1

        if not records:
            raise RuntimeError(
                "No VQA records could be matched to images. Confirm that "
                f"'{self.image_dir}' contains the COCO val2014 JPG files."
            )

        # A deterministic vocabulary keeps checkpoint class indices reproducible.
        if max_answers is not None:
            ranked = sorted(answer_counts.items(), key=lambda item: (-item[1], item[0]))
            answers = sorted(answer for answer, _ in ranked[:max_answers])
            keep = set(answers)
            records = [record for record in records if record["answer"] in keep]
        else:
            answers = sorted(answer_counts)
        self.answer2idx = {answer: index for index, answer in enumerate(answers)}
        self.idx2answer = {index: answer for answer, index in self.answer2idx.items()}
        self.records = records

    @staticmethod
    def _require_path(path: Path, description: str) -> None:
        if not path.exists():
            raise FileNotFoundError(f"{description} was not found: {path}")

    @staticmethod
    def normalize_answer(answer: object) -> str:
        """Apply the basic normalization used by the released VQA annotations."""
        return " ".join(str(answer).lower().strip().split())

    def _image_path(self, image_id: int) -> Path | None:
        """Support both official COCO filenames and bare numeric image names."""
        split = self.image_dir.name
        candidates = (
            self.image_dir / f"COCO_{split}_{image_id:012d}.jpg",
            self.image_dir / f"{image_id:012d}.jpg",
            self.image_dir / f"{image_id}.jpg",
        )
        return next((path for path in candidates if path.is_file()), None)

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int) -> tuple[str, str, torch.Tensor]:
        record = self.records[index]
        answer = str(record["answer"])
        return (
            str(record["image_path"]),
            str(record["question"]),
            torch.tensor(self.answer2idx[answer], dtype=torch.long),
        )
