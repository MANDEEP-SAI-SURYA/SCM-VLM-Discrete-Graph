"""Answer questions directly from a (factual or counterfactual) scene graph.

The SCM-VLM classifier gives a distribution over the VQA answer vocabulary.
For what-if questions it also helps to read the answer off the intervened
scene itself: after do(bicycle := motorcycle) the object the person is
riding *is* a motorcycle.  This module does that for common question types,
using only the detected labels, boxes and intervened attributes, so it works
for any image:

* counting          "how many cars are there?"
* existence         "is there a dog?"
* colour/attribute  "what color is the traffic light?", "is the light red?"
* relations         "what is the man riding / holding / sitting on / next to?"
* scene listing     "what is in the picture?"

It returns ``None`` when the question type is not covered, so the caller can
fall back to the classifier or to the LLM.
"""

from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Callable


@dataclass
class SceneObject:
    index: int
    label: str
    box: tuple[float, float, float, float] | None = None   # x0, y0, x1, y1
    attributes: list[str] = field(default_factory=list)


@dataclass
class SceneAnswer:
    answer: str
    reason: str
    question_type: str


COLOURS = {"red", "green", "yellow", "amber", "orange", "blue", "white", "black", "grey", "gray",
           "brown", "pink", "purple", "silver", "gold"}

# Generic (COCO-level) knowledge about what a verb can apply to; not tied to any image.
_VERB_OBJECTS = {
    "riding": {"bicycle", "motorcycle", "horse", "elephant", "skateboard", "surfboard", "skis",
               "snowboard", "bus", "train", "boat", "car", "truck"},
    "driving": {"car", "bus", "truck", "motorcycle", "train", "boat", "airplane"},
    "holding": {"umbrella", "cell phone", "cup", "bottle", "wine glass", "book", "remote", "knife",
                "fork", "spoon", "tennis racket", "baseball bat", "baseball glove", "frisbee",
                "kite", "handbag", "suitcase", "toothbrush", "scissors", "hair drier", "banana",
                "apple", "sandwich", "donut", "pizza", "hot dog", "cake", "teddy bear", "sports ball",
                "skateboard", "surfboard", "laptop", "mouse", "keyboard", "vase", "backpack"},
    "carrying": {"umbrella", "handbag", "backpack", "suitcase", "bag", "surfboard", "skis",
                 "snowboard", "skateboard", "book", "bottle", "cup", "kite", "laptop"},
    "wearing": {"tie", "backpack", "handbag"},
    "eating": {"banana", "apple", "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza",
               "donut", "cake"},
    "using": {"cell phone", "laptop", "keyboard", "mouse", "remote", "umbrella", "tennis racket",
              "baseball bat", "toothbrush", "scissors"},
    "sitting on": {"chair", "couch", "bench", "bed", "toilet", "bicycle", "motorcycle", "horse"},
    "standing on": {"skateboard", "surfboard", "skis", "snowboard", "bench", "boat"},
    "lying on": {"bed", "couch", "bench"},
    "pushing": {"bicycle", "motorcycle", "suitcase", "car"},
    "walking": {"dog", "horse"},
}
# objects you ride on top of are preferred over vehicles you ride inside
_RIDE_ON_TOP = {"bicycle", "motorcycle", "horse", "elephant", "skateboard", "surfboard", "skis", "snowboard"}
# verbs / prepositions that need physical contact between the two boxes
_CONTACT = {"riding", "driving", "holding", "carrying", "wearing", "eating", "using", "sitting on",
            "standing on", "lying on", "pushing", "on", "riding on"}
# verbs where the object supports the subject, so it should be lower in the image
_SUPPORT = {"riding", "sitting on", "standing on", "lying on", "on", "riding on"}

_VERB_ALIASES = {
    "ride": "riding", "rides": "riding", "riding on": "riding", "on": None, "near": None,
    "next to": None, "beside": None, "behind": None, "under": None, "in front of": None,
    "hold": "holding", "holds": "holding", "carry": "carrying", "carries": "carrying",
    "wear": "wearing", "wears": "wearing", "eat": "eating", "eats": "eating", "use": "using",
    "uses": "using", "drive": "driving", "drives": "driving", "sitting": "sitting on",
    "sit on": "sitting on", "sits on": "sitting on", "standing": "standing on",
    "stand on": "standing on", "lying": "lying on", "push": "pushing", "walk": "walking",
}
_VERB_PATTERN = (
    r"riding on|riding|ride|rides|driving|drive|drives|holding|hold|holds|carrying|carry|carries|"
    r"wearing|wear|wears|eating|eat|eats|using|use|uses|sitting on|sits on|sit on|sitting|"
    r"standing on|stand on|standing|lying on|lying|pushing|push|walking|walk|"
    r"next to|in front of|beside|behind|under|near|on"
)


def _iou_parts(a, b):
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))
    area_b = max(1e-6, (b[2] - b[0]) * (b[3] - b[1]))
    return inter, area_a, area_b


def _relation_score(subject: SceneObject, candidate: SceneObject, verb: str | None,
                    verb_text: str) -> float | None:
    """Geometric plausibility of '<subject> <verb> <candidate>'.  Returns None when the
    relation is physically impossible (contact verb but the boxes do not touch)."""
    if subject.box is None or candidate.box is None:
        # e.g. an object added by do(): its position is unknown, so only the
        # verb prior can support it (scored below any object that touches).
        return 0.1
    inter, area_s, area_c = _iou_parts(subject.box, candidate.box)
    contact = inter > 0
    if (verb in _CONTACT or verb_text in _CONTACT) and not contact:
        return None
    score = 2.0 * inter / min(area_s, area_c)                     # overlap / containment
    cs = ((subject.box[0] + subject.box[2]) / 2, (subject.box[1] + subject.box[3]) / 2)
    cc = ((candidate.box[0] + candidate.box[2]) / 2, (candidate.box[1] + candidate.box[3]) / 2)
    scale = max(subject.box[2] - subject.box[0], subject.box[3] - subject.box[1], 1e-6)
    distance = ((cs[0] - cc[0]) ** 2 + (cs[1] - cc[1]) ** 2) ** 0.5 / scale
    score += max(0.0, 1.0 - distance)                              # proximity
    if verb in _SUPPORT or verb_text in _SUPPORT:
        score += 0.5 if cc[1] > cs[1] else -0.5                      # support is below the subject
    if area_c > 4 * area_s and inter >= 0.9 * area_s:
        score -= 1.0                                                 # subject is *inside* it
    return score


def _plural(label: str, count: int) -> str:
    if count == 1:
        return label
    if label.endswith(("s", "x", "ch", "sh")):
        return label + "es"
    if label == "person":
        return "people"
    return label + "s"


def answer_from_scene(
    question: str,
    objects: list[SceneObject],
    resolve: Callable[[str], str | None],
) -> SceneAnswer | None:
    """``resolve`` maps a phrase ("the man", "cars") to a detector label or None."""
    q = " ".join(question.lower().strip().rstrip("?.!").split())
    labels = [obj.label for obj in objects]

    def instances(phrase: str) -> tuple[str | None, list[SceneObject]]:
        label = resolve(phrase)
        return label, [obj for obj in objects if label is not None and obj.label == label]

    # ---- counting ---------------------------------------------------------
    if (m := re.match(r"how many (?P<x>.+?)(?: are| is| can| do| does| in| on| at| there|$)", q)):
        label, found = instances(m.group("x"))
        if label is None:
            return None
        return SceneAnswer(str(len(found)), f"{len(found)} {_plural(label, len(found))} in the scene", "count")

    # ---- existence --------------------------------------------------------
    if (m := re.match(r"(?:is|are) there (?:a |an |any |some |the )?(?P<x>.+?)(?: in| on| at| near|$)", q)):
        label, found = instances(m.group("x"))
        if label is None:
            return None
        return SceneAnswer("yes" if found else "no",
                           f"{len(found)} {_plural(label, len(found))} in the scene", "existence")

    # ---- colour / attribute value ------------------------------------------
    if (m := re.match(r"what colou?r (?:is|are) (?:the |this |that )?(?P<x>.+)$", q)):
        label, found = instances(m.group("x"))
        if label is None:
            return None
        if not found:
            return SceneAnswer("none", f"there is no {label} in the scene", "attribute")
        values = [a for obj in found for a in obj.attributes if a in COLOURS]
        if values:
            value = Counter(values).most_common(1)[0][0]
            return SceneAnswer(value, f"the {label} was set to {value} by the intervention", "attribute")
        return SceneAnswer("unknown", "colour is not encoded in the detected object labels", "attribute")

    # ---- yes/no attribute: "is the light red" ------------------------------
    colour_words = "|".join(sorted(COLOURS))
    if (m := re.match(rf"(?:is|are) (?:the |this |that )?(?P<x>.+?) (?P<c>{colour_words})$", q)):
        label, found = instances(m.group("x"))
        if label is None:
            return None
        if not found:
            return SceneAnswer("no", f"there is no {label} in the scene", "attribute")
        values = {a for obj in found for a in obj.attributes if a in COLOURS}
        if values:
            answer = "yes" if m.group("c") in values else "no"
            return SceneAnswer(answer, f"the {label} is {', '.join(sorted(values))}", "attribute")
        return SceneAnswer("unknown", "colour is not encoded in the detected object labels", "attribute")

    # ---- relations: "what is the man riding" --------------------------------
    if (m := re.match(rf"what (?:is|are|was|were) (?:the |a |this |that )?(?P<s>.+?) (?P<v>{_VERB_PATTERN})"
                      rf"(?: (?:on|in|with))?$", q)) or \
       (m := re.match(rf"what (?:does|do|did) (?:the |a |this |that )?(?P<s>.+?) (?P<v>{_VERB_PATTERN})$", q)):
        subject_label, subjects = instances(m.group("s"))
        if subject_label is None:
            return None
        if not subjects:
            return SceneAnswer("nothing", f"there is no {subject_label} in the scene", "relation")
        verb_text = m.group("v")
        verb = _VERB_ALIASES.get(verb_text, verb_text)
        candidates = [obj for obj in objects if obj.label != subject_label]
        if verb in _VERB_OBJECTS:
            candidates = [obj for obj in candidates if obj.label in _VERB_OBJECTS[verb]]
            if verb == "riding" and any(obj.label in _RIDE_ON_TOP for obj in candidates):
                candidates = [obj for obj in candidates if obj.label in _RIDE_ON_TOP]
        scored = []
        for candidate in candidates:
            scores = [score for s in subjects
                      if (score := _relation_score(s, candidate, verb, verb_text)) is not None]
            if scores:
                scored.append((max(scores), candidate))
        what = f"{verb or verb_text}"
        if not scored:
            return SceneAnswer("nothing", f"no object in the scene that the {subject_label} could be {what}",
                               "relation")
        score, obj = max(scored, key=lambda item: item[0])
        name = " ".join(obj.attributes + [obj.label]) if obj.attributes else obj.label
        return SceneAnswer(name, f"{subject_label} {verb_text} {name} (node {obj.index}, score {score:.2f})",
                           "relation")

    # ---- what is in the picture --------------------------------------------
    if re.match(r"what (?:is|are) (?:in|there in|shown in|visible in) (?:the |this )?(?:picture|image|photo|scene)", q) \
            or q in {"what is in the picture", "what is in the image", "what do you see"}:
        counts = Counter(labels)
        if not counts:
            return SceneAnswer("nothing", "no objects", "scene")
        text = ", ".join(f"{n} {_plural(label, n)}" for label, n in counts.most_common())
        return SceneAnswer(counts.most_common(1)[0][0], text, "scene")

    return None
