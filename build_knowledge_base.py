"""Build the symbolic class-pair knowledge base used by --knowledge-prior (models/knowledge.py).

    python build_knowledge_base.py --coco-instances data/annotations/instances_train2014.json \
        --out checkpoints/knowledge_base.pt [--external my_rules.tsv] [--exclude-val]

--exclude-val removes the images of the fixed validation split of train.py (recommended when
the knowledge base is used for the experiments).  Prints the strongest symbolic triples.
"""

from __future__ import annotations

import argparse

import torch

from datasets.counterfactual import load_coco_instances
from models.knowledge import (KNOWLEDGE_RELATIONS, KNOWLEDGE_VERSION, build_knowledge_base,
                              knowledge_triples, load_external_triples)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--coco-instances", required=True)
    parser.add_argument("--out", default="checkpoints/knowledge_base.pt")
    parser.add_argument("--external", default=None, help="TSV: subject<TAB>relation<TAB>object<TAB>confidence")
    parser.add_argument("--exclude-val", action="store_true",
                        help="exclude the validation images of train.py's fixed split")
    parser.add_argument("--show", type=int, default=25, help="number of triples to print")
    args = parser.parse_args(argv)

    exclude = set()
    if args.exclude_val:
        import train
        from datasets.datasetBuilder import VQACOCO

        dataset = VQACOCO(train.IMAGE_DIR, train.QUESTION_FILE, train.ANNOTATION_FILE, max_answers=train.MAX_ANSWERS)
        _, val_idx = train.split_by_image(dataset, train.TRAIN_SAMPLES, train.VAL_FRACTION, train.SPLIT_SEED)
        exclude = {int(dataset.records[i]["image_id"]) for i in val_idx}
    ground_truth, _ = load_coco_instances(args.coco_instances)
    knowledge = build_knowledge_base(ground_truth, exclude_image_ids=exclude)
    if args.external:
        knowledge, added = load_external_triples(args.external, knowledge)
        print(f"added {added} external triples")
    torch.save({"knowledge": knowledge, "relations": KNOWLEDGE_RELATIONS, "version": KNOWLEDGE_VERSION,
                "excluded_images": len(exclude)}, args.out)
    print(f"saved {tuple(knowledge.shape)} knowledge base to {args.out} "
          f"({len(ground_truth) - len(exclude & set(ground_truth))} images)")
    for subject, relation, obj, confidence in knowledge_triples(knowledge, 0.5, top=args.show):
        print(f"  ({subject}, {relation}, {obj})  {confidence:.2f}")


if __name__ == "__main__":
    main()
