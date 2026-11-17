"""Download everything train.py needs straight from the official sites, without the 13 GB
COCO zip. Use it on Kaggle, a laptop, or any machine where a ready-made dataset isn't available.

    python download_data.py --images 8000

It downloads:
  1. VQA v2 training questions + answers   -> coco/vqa/            (~30 MB)
  2. COCO 2014 instance annotations         -> coco/annotations/    (~240 MB zip; train json kept)
  3. a FIXED random subset of COCO train2014 images (--images, default 8000, ~1.3 GB)
                                            -> coco/train2014/train2014/

The image subset is chosen with a fixed seed from the images that have VQA questions, so every
run (soft / discrete / future-work variants, all seeds) uses exactly the same images and the
same train / validation split. 8000 images hold about 43,000 questions, enough for
`train.py --samples 30000`. Images that are already on disk are skipped, so an interrupted
download can simply be restarted.
"""

from __future__ import annotations

import argparse
import json
import random
import shutil
import sys
import time
import urllib.request
import zipfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

VQA_QUESTIONS = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Questions_Train_mscoco.zip"
VQA_ANSWERS = "https://s3.amazonaws.com/cvmlp/vqa/mscoco/vqa/v2_Annotations_Train_mscoco.zip"
COCO_ANNOTATIONS = "http://images.cocodataset.org/annotations/annotations_trainval2014.zip"
COCO_IMAGES = "http://images.cocodataset.org/train2014"

QUESTION_FILE = "v2_OpenEnded_mscoco_train2014_questions.json"
ANSWER_FILE = "v2_mscoco_train2014_annotations.json"
INSTANCES_FILE = "instances_train2014.json"


def fetch(url: str, target: Path, retries: int = 4, timeout: int = 60) -> None:
    """Download url to target (atomically, with retries)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response, partial.open("wb") as out:
                shutil.copyfileobj(response, out, length=1 << 20)
            partial.replace(target)
            return
        except Exception as error:  # network hiccup: retry with back-off
            if attempt == retries - 1:
                raise RuntimeError(f"download failed: {url} ({error})") from error
            time.sleep(2 ** attempt)


def fetch_zip_member(url: str, member_name: str, out_dir: Path, work: Path) -> Path:
    """Download a zip (unless the wanted file already exists) and extract one file from it."""
    target = out_dir / member_name
    if target.is_file():
        print(f"  {target} already exists")
        return target
    archive = work / Path(url).name
    if not archive.is_file():
        print(f"  downloading {url}")
        fetch(url, archive, timeout=300)
    with zipfile.ZipFile(archive) as zf:
        member = next(n for n in zf.namelist() if n.endswith(member_name))
        out_dir.mkdir(parents=True, exist_ok=True)
        with zf.open(member) as src, target.open("wb") as dst:
            shutil.copyfileobj(src, dst, length=1 << 20)
    archive.unlink()
    print(f"  extracted {target}")
    return target


def choose_images(question_file: Path, count: int, seed: int) -> list[int]:
    with question_file.open("r", encoding="utf-8") as file:
        image_ids = sorted({int(q["image_id"]) for q in json.load(file)["questions"]})
    count = min(count, len(image_ids))
    return sorted(random.Random(seed).sample(image_ids, count))


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", default="coco", help="data folder used by train.py (default: coco)")
    parser.add_argument("--images", type=int, default=8000, help="number of COCO train2014 images")
    parser.add_argument("--seed", type=int, default=0, help="seed of the image subset (keep it fixed)")
    parser.add_argument("--workers", type=int, default=32, help="parallel image downloads")
    parser.add_argument("--image-base-url", default=COCO_IMAGES, help=argparse.SUPPRESS)
    parser.add_argument("--vqa-questions-url", default=VQA_QUESTIONS, help=argparse.SUPPRESS)
    parser.add_argument("--vqa-answers-url", default=VQA_ANSWERS, help=argparse.SUPPRESS)
    parser.add_argument("--coco-annotations-url", default=COCO_ANNOTATIONS, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)

    root = Path(args.root)
    work = root / "_downloads"
    work.mkdir(parents=True, exist_ok=True)

    print("1/3 VQA v2 questions and answers")
    questions = fetch_zip_member(args.vqa_questions_url, QUESTION_FILE, root / "vqa", work)
    fetch_zip_member(args.vqa_answers_url, ANSWER_FILE, root / "vqa", work)

    print("2/3 COCO instance annotations (needed for counterfactual examples and the knowledge base)")
    fetch_zip_member(args.coco_annotations_url, INSTANCES_FILE, root / "annotations", work)

    print(f"3/3 {args.images} COCO train2014 images (fixed subset, seed {args.seed})")
    image_dir = root / "train2014" / "train2014"
    image_dir.mkdir(parents=True, exist_ok=True)
    ids = choose_images(questions, args.images, args.seed)
    todo = [i for i in ids if not (image_dir / f"COCO_train2014_{i:012d}.jpg").is_file()]
    print(f"  {len(ids) - len(todo)} already on disk, {len(todo)} to download")
    failed = []

    def one(image_id: int) -> None:
        name = f"COCO_train2014_{image_id:012d}.jpg"
        fetch(f"{args.image_base_url}/{name}", image_dir / name)

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(one, i): i for i in todo}
        for done, future in enumerate(as_completed(futures), start=1):
            if future.exception() is not None:
                failed.append(futures[future])
            if done % 500 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)} images", flush=True)
    shutil.rmtree(work, ignore_errors=True)
    (root / "subset_image_ids.json").write_text(json.dumps({"seed": args.seed, "image_ids": ids}))

    if failed:
        print(f"WARNING: {len(failed)} images failed; run the same command again to retry them.")
        return 1
    print(f"Done: {len(ids)} images in {image_dir}. Next: python validate_dataset.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
