"""COCO / DETR class ids (facebook/detr-resnet-50 ``config.id2label``).

COCO 2014 ``category_id`` values use the same ids, so ground-truth annotations
and DETR detections can be compared directly.  Index 91 is DETR's "no object"
logit; it is also used here as the id for unknown label names.
"""

DETR_ID2LABEL = {
    0: "N/A", 1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "airplane", 6: "bus",
    7: "train", 8: "truck", 9: "boat", 10: "traffic light", 11: "fire hydrant", 12: "N/A",
    13: "stop sign", 14: "parking meter", 15: "bench", 16: "bird", 17: "cat", 18: "dog",
    19: "horse", 20: "sheep", 21: "cow", 22: "elephant", 23: "bear", 24: "zebra", 25: "giraffe",
    26: "N/A", 27: "backpack", 28: "umbrella", 29: "N/A", 30: "N/A", 31: "handbag", 32: "tie",
    33: "suitcase", 34: "frisbee", 35: "skis", 36: "snowboard", 37: "sports ball", 38: "kite",
    39: "baseball bat", 40: "baseball glove", 41: "skateboard", 42: "surfboard",
    43: "tennis racket", 44: "bottle", 45: "N/A", 46: "wine glass", 47: "cup", 48: "fork",
    49: "knife", 50: "spoon", 51: "bowl", 52: "banana", 53: "apple", 54: "sandwich",
    55: "orange", 56: "broccoli", 57: "carrot", 58: "hot dog", 59: "pizza", 60: "donut",
    61: "cake", 62: "chair", 63: "couch", 64: "potted plant", 65: "bed", 66: "N/A",
    67: "dining table", 68: "N/A", 69: "N/A", 70: "toilet", 71: "N/A", 72: "tv", 73: "laptop",
    74: "mouse", 75: "remote", 76: "keyboard", 77: "cell phone", 78: "microwave", 79: "oven",
    80: "toaster", 81: "sink", 82: "refrigerator", 83: "N/A", 84: "book", 85: "clock",
    86: "vase", 87: "scissors", 88: "teddy bear", 89: "hair drier", 90: "toothbrush",
}

NO_OBJECT_ID = 91
NUM_CLASS_IDS = 92                       # size of the class-embedding table

LABEL2ID = {label: idx for idx, label in DETR_ID2LABEL.items() if label != "N/A"}
COCO_LABELS = sorted(LABEL2ID)            # the 80 real classes


def label_to_id(label: str) -> int:
    """DETR / COCO id of a class name (unknown names map to NO_OBJECT_ID)."""
    return LABEL2ID.get(str(label).lower(), NO_OBJECT_ID)


_IRREGULAR_PLURALS = {
    "person": "people", "mouse": "mice", "knife": "knives", "skis": "skis", "sheep": "sheep",
    "scissors": "scissors", "bus": "buses", "couch": "couches", "bench": "benches",
    "sandwich": "sandwiches", "toothbrush": "toothbrushes", "wine glass": "wine glasses",
    "tv": "tvs", "hair drier": "hair driers", "broccoli": "broccoli",
}


def plural(label: str) -> str:
    if label in _IRREGULAR_PLURALS:
        return _IRREGULAR_PLURALS[label]
    if label.endswith(("s", "x", "ch", "sh")):
        return label + "es"
    return label + "s"


def with_article(label: str) -> str:
    return ("an " if label[0] in "aeiou" else "a ") + label
