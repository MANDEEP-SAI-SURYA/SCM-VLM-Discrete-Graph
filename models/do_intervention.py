"""Pearl-style do() interventions for "what if" questions on SCM-VLM.

Graph convention used by the model
----------------------------------
``CausalAttentionBlock`` computes ``messages = A @ h``, i.e.

    h_i  <-  sum_j A[i, j] * W h_j

so ``A[i, j] > 0`` means object ``j`` is a *parent* of object ``i`` (edge
j -> i).  Row ``i`` holds the parents of ``i``; column ``j`` holds the
children of ``j``.

The intervention do(X_k = x') therefore:

1. cuts every incoming edge of k        ->  A[k, :] = 0          (paper Eq. 12)
2. sets the node value to x'            ->  h_k <- Embed(x')     (paper Eq. 13)
3. keeps every other structural equation (all other rows of A) fixed
4. recomputes only the descendants of k (paper Eq. 14); all other nodes keep
   their factual representation
5. decodes the counterfactual answer Y_do(X_k = x')                (paper Eq. 15)

Supported interventions
-----------------------
* ``remove``  - "what if there were no <X>":  the object(s) are deleted from
  the SCM; their children lose that parent (rows are renormalised over the
  remaining parents, matching the model's averaging mechanism).
* ``replace`` - "what if the <X> were a <Y>":  do(X_k = x') where x' is the
  object's own DETR embedding steered until DETR's class head reads <Y>.  The
  object keeps its position/appearance features and only its class changes.
* ``attribute`` - "what if the traffic light turns green":  do(X_k = x')
  where x' is the node value moved along a text-defined attribute direction
  (see ``DoInterventionReasoner._attribute``).  The object keeps its class.
* ``add``     - "what if there were a <Y>":  a new exogenous (parent-less)
  node is inserted; its outgoing edges are scored by the model's learned
  question-conditioned edge function.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F

from configs.coco_labels import NO_OBJECT_ID, label_to_id
from models.scene_answer import SceneObject


# =====================================================================
# Label resolution (free text -> DETR COCO label)
# =====================================================================

# Synonyms for the 80 COCO classes DETR detects (generic, not tied to any image).
# Exact class names and simple plurals are matched automatically; this table
# only adds everyday words people use in questions.
_LABEL_ALIASES = {
    "person": {"people", "persons", "man", "men", "woman", "women", "boy", "boys", "girl", "girls",
               "child", "children", "kid", "kids", "baby", "pedestrian", "pedestrians", "rider",
               "riders", "cyclist", "cyclists", "biker", "human", "humans", "guy", "guys", "lady",
               "ladies", "player", "players", "skier", "surfer", "skateboarder", "driver", "he",
               "she", "someone", "somebody", "worker", "officer", "crowd"},
    "bicycle": {"bike", "bikes", "cycle", "bicycles"},
    "car": {"vehicle", "vehicles", "taxi", "taxis", "cab", "cabs", "suv", "van", "sedan", "automobile"},
    "motorcycle": {"motorbike", "motorbikes", "scooter", "scooters", "moped"},
    "airplane": {"plane", "planes", "aeroplane", "jet", "aircraft"},
    "bus": {"busses", "coach"},
    "train": {"locomotive", "tram", "subway"},
    "truck": {"lorry", "pickup"},
    "boat": {"ship", "ships", "canoe", "kayak", "sailboat", "yacht"},
    "traffic light": {"signal", "signals", "traffic signal", "stoplight", "stop light", "light",
                      "lights", "signal light", "traffic lights"},
    "fire hydrant": {"hydrant"},
    "stop sign": {"sign", "signs"},
    "parking meter": {"meter"},
    "bench": {"benches", "park bench"},
    "bird": {"pigeon", "seagull", "duck", "parrot"},
    "cat": {"kitten", "kitty"},
    "dog": {"puppy", "puppies"},
    "horse": {"pony", "horses"},
    "sheep": {"lamb", "lambs"},
    "cow": {"cattle", "bull", "calf"},
    "elephant": {"elephants"},
    "bear": {"teddy"},
    "zebra": {"zebras"},
    "giraffe": {"giraffes"},
    "backpack": {"rucksack", "bag pack"},
    "umbrella": {"parasol"},
    "handbag": {"purse", "bag", "bags"},
    "tie": {"necktie"},
    "suitcase": {"luggage", "suitcases"},
    "frisbee": {"disc"},
    "skis": {"ski"},
    "snowboard": {"snowboards"},
    "sports ball": {"ball", "balls", "football", "soccer ball", "basketball", "tennis ball"},
    "kite": {"kites"},
    "baseball bat": {"bat"},
    "baseball glove": {"glove", "mitt"},
    "skateboard": {"skate board"},
    "surfboard": {"surf board", "board"},
    "tennis racket": {"racket", "racquet", "tennis racquet"},
    "bottle": {"bottles"},
    "wine glass": {"glass", "glasses", "wineglass"},
    "cup": {"mug", "mugs", "cups"},
    "fork": {"forks"},
    "knife": {"knives"},
    "spoon": {"spoons"},
    "bowl": {"bowls"},
    "banana": {"bananas"},
    "apple": {"apples"},
    "sandwich": {"sandwiches", "burger"},
    "orange": {"oranges"},
    "broccoli": {"broccolis"},
    "carrot": {"carrots"},
    "hot dog": {"hotdog", "hotdogs"},
    "pizza": {"pizzas"},
    "donut": {"doughnut", "doughnuts", "donuts"},
    "cake": {"cakes"},
    "chair": {"chairs", "seat", "seats"},
    "couch": {"sofa", "sofas", "couches"},
    "potted plant": {"plant", "plants", "flower pot", "pot plant"},
    "bed": {"beds"},
    "dining table": {"table", "tables", "desk"},
    "toilet": {"toilets", "wc"},
    "tv": {"television", "tvs", "televisions", "monitor", "screen"},
    "laptop": {"computer", "notebook", "laptops"},
    "mouse": {"computer mouse"},
    "remote": {"remote control", "controller"},
    "keyboard": {"keyboards"},
    "cell phone": {"phone", "phones", "mobile", "mobile phone", "smartphone", "cellphone"},
    "microwave": {"microwave oven"},
    "oven": {"stove", "ovens"},
    "toaster": {"toasters"},
    "sink": {"sinks", "basin"},
    "refrigerator": {"fridge", "fridges"},
    "book": {"books"},
    "clock": {"clocks", "watch"},
    "vase": {"vases"},
    "scissors": {"scissor"},
    "teddy bear": {"teddy", "stuffed animal", "plush"},
    "hair drier": {"hair dryer", "hairdryer", "blow dryer"},
    "toothbrush": {"toothbrushes", "tooth brush"},
}
for _label, _words in _LABEL_ALIASES.items():
    _words.add(_label)

_ARTICLES = re.compile(r"^(?:the|a|an|any|some|another|all|all the|those|these|that|this)\s+")


def _singular_forms(word: str) -> list[str]:
    """Possible singular forms ('buses' -> bus, 'horses' -> horse, 'berries' -> berry)."""
    forms = []
    if word.endswith("ies") and len(word) > 4:
        forms.append(word[:-3] + "y")
    if word.endswith("es") and len(word) > 3:
        forms.append(word[:-2])
    if word.endswith("s") and not word.endswith("ss") and len(word) > 2:
        forms.append(word[:-1])
    return forms


def _singular(word: str) -> str:
    forms = _singular_forms(word)
    return forms[-1] if forms else word


def resolve_label(phrase: str, known_labels: list[str]) -> str | None:
    """Map a free-text object phrase onto one of the detector's label names."""
    text = re.sub(r"[^a-z ]", " ", phrase.lower())
    text = re.sub(r"\s+", " ", text).strip()
    while True:
        stripped = _ARTICLES.sub("", text)
        if stripped == text:
            break
        text = stripped
    if not text:
        return None

    known = {label.lower(): label for label in known_labels}
    words = text.split()
    candidates = [text, *_singular_forms(text)]
    # also try the last one/two words ("the red car" -> "car")
    last_two = " ".join(words[-2:])
    candidates += [last_two, *_singular_forms(last_two), words[-1], *_singular_forms(words[-1])]

    for candidate in candidates:
        if candidate in known:
            return known[candidate]
    for candidate in candidates:
        for label, aliases in _LABEL_ALIASES.items():
            if candidate in aliases and label in known:
                return known[label]
    return None


# =====================================================================
# Intervention specification + "what if" parser
# =====================================================================

@dataclass
class Intervention:
    kind: str                      # "remove" | "replace" | "attribute" | "add"
    target: str | None = None      # object phrase being intervened on
    new_label: str | None = None   # new class (replace/add) or attribute (attribute)
    nodes: list[int] | None = None  # optional explicit node indices

    def describe(self, mode: str = "do") -> str:
        target = self.target or "node " + ",".join(f"#{i}" for i in (self.nodes or []))
        if mode == "recompute":
            target = re.sub(r"^(?:the|a|an)\s+", "", target)
            if self.kind == "remove":
                return f"remove {target}"
            if self.kind == "replace":
                return f"replace {target} with {self.new_label}"
            if self.kind == "attribute":
                return f"make {target} {self.new_label}"
            return f"add {self.new_label}"
        if self.kind == "remove":
            return f"do(remove {target})"
        if self.kind == "replace":
            return f"do({target} := {self.new_label})"
        if self.kind == "attribute":
            return f"do({target}.attribute := {self.new_label})"
        return f"do(add {self.new_label})"

    def default_question(self) -> str | None:
        """Question to ask in the intervened world when the user gives none."""
        target = (self.target or "object").strip()
        target = re.sub(r"^(?:the|a|an)\s+", "", target)
        if self.kind == "attribute":
            if (self.new_label or "").lower() in _COLOURS:
                return f"What color is the {target}?"
            return f"Is the {target} {self.new_label}?"
        return None


_REMOVE_PATTERNS = [
    r"there (?:were|was|is|are) no (?P<x>.+)",
    r"there (?:were|was|is|are)n'?t any (?P<x>.+)",
    r"(?:we |you |i )?remove(?:d)? (?P<x>.+)",
    r"(?P<x>.+?) (?:were|was|is|are|got|had been|has been|have been) (?:removed|gone|absent|missing|not there|taken away|deleted)",
    r"(?P<x>.+?) (?:did not|didn't|does not|doesn't) exist",
    r"without (?P<x>.+)",
    r"no (?P<x>.+)",
]
_REPLACE_PATTERNS = [
    r"(?:we |you |i )?replace(?:d)? (?P<x>.+?) (?:with|by) (?P<y>.+)",
    r"(?P<x>.+?) (?:were|was|is|are) replaced (?:with|by) (?P<y>.+)",
    r"(?!there\b)(?P<x>.+?) (?:were|was|is|are|became|becomes|turned into|changed to|changed into) (?:a|an) (?P<y>.+)",
    r"(?!there\b)(?P<x>.+?) (?:turned into|turns into|turn into|changed into|changes into) (?P<y>.+)",
    r"instead of (?P<x>.+?),? there (?:were|was|is) (?:a |an )?(?P<y>.+)",
]
_COLOURS = {"red", "green", "yellow", "amber", "orange", "blue", "white", "black", "grey", "gray",
            "brown", "pink", "purple", "silver", "gold"}

# "the traffic light turns green", "the car was blue", "the light changed to red",
# "the umbrella were closed".  If the new value is actually an object class
# (e.g. "the bicycle became motorcycle") the reasoner turns it into a replace.
_ATTRIBUTE_PATTERNS = [
    r"(?!there\b)(?P<x>.+?) (?:turns|turned|turn|goes|went|go|gets|got|get|becomes|became|become"
    r"|changes to|changed to|change to|switches to|switched to|switch to|were painted|was painted"
    r"|is painted|are painted|were|was|is|are|looks|looked|shows|showed) "
    r"(?:to |into )?(?P<y>[a-z]+(?: [a-z]+)?)",
]

_ADD_PATTERNS = [
    r"there (?:were|was|is|are) (?:also )?(?:a|an|another|some|one more|more) (?P<y>.+)",
    r"(?:we |you |i )?add(?:ed)? (?:a |an |another |some )?(?P<y>.+)",
    r"(?:a|an) (?P<y>.+?) (?:appeared|was added|were added|came)",
]


def _match(patterns: list[str], clause: str):
    for pattern in patterns:
        match = re.fullmatch(pattern, clause)
        if match:
            return match
    return None


def parse_whatif(text: str) -> tuple[Intervention, str | None]:
    """Split "What if <clause>, <question>?" into an Intervention + question.

    Examples
    --------
    "What if there were no umbrella, what is the person holding?"
    "What if the bicycle were a motorcycle? What is the man riding?"
    "What if we removed the cars - is the road busy?"
    "What if there was a dog?"            (question supplied separately)
    """
    raw = " ".join(text.strip().split())
    lowered = raw.lower()
    match = re.match(r"^\s*(?:but\s+)?(?:what|wat)\s+(?:if|would happen if|happens if)\s+(?P<rest>.+)$", lowered)
    if not match:
        raise ValueError("A what-if question must start with 'What if ...'.")
    rest = match.group("rest").strip()

    # split clause from follow-up question at the first ',', '?', ';', ' - ', or ' then '
    split = re.split(r"\s*(?:,|\?|;|\s-\s|\s+then\s+)\s*", rest, maxsplit=1)
    clause = split[0].strip(" .")
    question = split[1].strip() if len(split) > 1 and split[1].strip() else None
    if question and not question.endswith("?"):
        question = question.rstrip(".") + "?"

    if (m := _match(_REPLACE_PATTERNS, clause)):
        return Intervention("replace", target=m.group("x").strip(), new_label=m.group("y").strip()), question
    if (m := _match(_REMOVE_PATTERNS, clause)):
        return Intervention("remove", target=m.group("x").strip()), question
    if (m := _match(_ADD_PATTERNS, clause)):
        return Intervention("add", new_label=m.group("y").strip()), question
    if (m := _match(_ATTRIBUTE_PATTERNS, clause)):
        if re.match(r"(?:not|no longer|never)\b", m.group("y")) or re.search(r"\b(?:not|n't)$", m.group("x")):
            raise ValueError(
                f"'{clause}' is a negation, which does not fix a single value for do(). "
                "Say the new value instead, e.g. 'what if the traffic light turns green'."
            )
        return Intervention("attribute", target=m.group("x").strip(), new_label=m.group("y").strip()), question
    raise ValueError(
        f"Could not understand the intervention '{clause}'. Try e.g. 'what if there were no car', "
        "'what if the bicycle were a motorcycle', 'what if the traffic light turns green', "
        "or 'what if there was a dog'."
    )


def parse_do_spec(spec: str) -> Intervention:
    """Parse an explicit spec: 'remove:car', 'replace:umbrella->bicycle', 'add:dog',
    'attribute:traffic light->green', or index form 'remove:#3', 'replace:#3->dog'."""
    kind, _, body = spec.partition(":")
    kind = kind.strip().lower()
    body = body.strip()

    def nodes_of(part: str):
        part = part.strip()
        if part.startswith("#"):
            return None, [int(i) for i in part[1:].split(",")]
        return part, None

    if kind == "remove":
        target, nodes = nodes_of(body)
        return Intervention("remove", target=target, nodes=nodes)
    if kind == "replace":
        left, _, right = body.partition("->")
        target, nodes = nodes_of(left)
        return Intervention("replace", target=target, new_label=right.strip(), nodes=nodes)
    if kind == "attribute":
        left, _, right = body.partition("->")
        target, nodes = nodes_of(left)
        return Intervention("attribute", target=target, new_label=right.strip(), nodes=nodes)
    if kind == "add":
        return Intervention("add", new_label=body)
    raise ValueError(f"Unknown intervention kind '{kind}' (use remove/replace/attribute/add).")


# =====================================================================
# Class steering in DETR embedding space  (x' for replace / add)
# =====================================================================

class ClassSteerer:
    """Moves a DETR decoder embedding until DETR's own class head predicts a
    target class.  This gives an x' that lives in the same space the model was
    trained on, and whose new class is verifiable."""

    def __init__(self, classifier: nn.Linear, label2id: dict[str, int]) -> None:
        self.classifier = classifier
        self.label2id = {k.lower(): v for k, v in label2id.items()}

    def labels(self) -> list[str]:
        return [label for label in self.label2id if label not in {"n/a", "no object"}]

    @torch.no_grad()
    def steer(self, base: torch.Tensor, target_label: str, min_prob: float = 0.6,
              max_steps: int = 300) -> tuple[torch.Tensor, float, bool]:
        target = self.label2id[target_label.lower()]
        weight = self.classifier.weight
        x = base.clone().float()
        step = 0.05 * x.norm().clamp(min=1.0)
        prob = 0.0
        for _ in range(max_steps):
            probs = F.softmax(self.classifier(x), dim=-1)
            prob = float(probs[target])
            if int(probs.argmax()) == target and prob >= min_prob:
                return x, prob, True
            competitor = probs.clone()
            competitor[target] = -1.0
            direction = weight[target] - weight[int(competitor.argmax())]
            x = x + step * direction / direction.norm().clamp(min=1e-8)
        return x, prob, prob >= 0.5


# =====================================================================
# Worlds
# =====================================================================

@dataclass
class World:
    labels: list[str]
    raw_objects: torch.Tensor      # [N, OBJECT_DIM]  (DETR space)
    objects: torch.Tensor          # [N, HIDDEN_DIM]  (projected node values)
    adjacency: torch.Tensor        # [N, N]  A[i, j] > 0  <=>  j -> i
    hidden: torch.Tensor           # [N, HIDDEN_DIM]  after causal layers
    logits: torch.Tensor           # [C]
    question: str
    question_embedding: torch.Tensor
    boxes: list | None = None
    scores: list | None = None
    edge_probs: torch.Tensor | None = None   # [N, N] sigmoid edge scores before top-k/threshold
    boxes_norm: torch.Tensor | None = None   # [N, 4] boxes scaled to 0-1 (for the relation graph)
    class_ids: torch.Tensor | None = None    # [N] DETR / COCO class ids (node identity)
    relation_probs: torch.Tensor | None = None  # [N, N, R] relation-type probabilities (R > 1)


@dataclass
class CounterfactualResult:
    intervention: Intervention
    factual: World
    counterfactual: World
    intervened_nodes: list[int]        # indices in the factual world (or new node for add)
    descendants: list[int]             # indices in the factual world
    unaffected: list[int]              # factual nodes whose representation is untouched
    non_descendant_max_drift: float    # sanity check: should be ~0
    steering: dict = field(default_factory=dict)
    cf_index_map: dict = field(default_factory=dict)  # factual idx -> cf idx
    mode: str = "do"                   # "do" (fixed graph) or "recompute" (graph re-estimated)


def descendants_of(adjacency: torch.Tensor, nodes: list[int], eps: float = 1e-8) -> list[int]:
    """All nodes reachable from ``nodes`` following j -> i edges (A[i, j] > 0)."""
    graph = adjacency.detach().cpu()
    seen: set[int] = set()
    frontier = list(nodes)
    while frontier:
        current = frontier.pop()
        children = torch.nonzero(graph[:, current] > eps, as_tuple=False).flatten().tolist()
        for child in children:
            if child not in seen and child not in nodes:
                seen.add(child)
                frontier.append(child)
    return sorted(seen)


def _renormalise_rows(adjacency: torch.Tensor) -> torch.Tensor:
    return adjacency / (adjacency.sum(dim=-1, keepdim=True) + 1e-8)


# =====================================================================
# Reasoner
# =====================================================================

class DoInterventionReasoner:
    MODES = ("recompute", "do")

    def __init__(self, model, idx2answer: dict[int, str] | None = None,
                 steerer: ClassSteerer | None = None, attribute_strength: float = 0.5,
                 mode: str = "recompute") -> None:
        """mode:
        * "recompute" - edit the scene (replace / remove / add / recolour an object), then let
          the model re-calculate every relation between the objects and answer the question
          with exactly the same pipeline as a normal question.
        * "do"        - strict SCM intervention do(X_k = x'): the learned graph is kept, only
          the parents of k are cut, and only the descendants of k are recomputed.
        """
        if mode not in self.MODES:
            raise ValueError(f"mode must be one of {self.MODES}")
        self.mode = mode
        self.model = model
        self.attribute_strength = attribute_strength
        self.idx2answer = idx2answer or {}
        if steerer is None:
            detr = getattr(getattr(model, "vision", None), "model", None)
            if detr is not None and hasattr(detr, "class_labels_classifier"):
                steerer = ClassSteerer(detr.class_labels_classifier, detr.config.label2id)
        self.steerer = steerer

    # ---------------------------------------------------------------
    @property
    def known_labels(self) -> list[str]:
        return self.steerer.labels() if self.steerer else []

    def answer_name(self, index: int) -> str:
        return self.idx2answer.get(int(index), f"class_{int(index)}")

    def top_answers(self, logits: torch.Tensor, k: int = 5) -> list[tuple[str, float]]:
        probs = F.softmax(logits, dim=-1)
        values, indices = probs.topk(min(k, probs.numel()))
        return [(self.answer_name(i), float(v)) for v, i in zip(values, indices)]

    # ---------------------------------------------------------------
    def _normalise(self, adjacency: torch.Tensor) -> torch.Tensor:
        graph = getattr(self.model, "graph", None)
        if graph is not None and hasattr(graph, "normalise"):
            return graph.normalise(adjacency)
        return _renormalise_rows(adjacency)

    @staticmethod
    def _batch_boxes(boxes_norm):
        return None if boxes_norm is None else boxes_norm.unsqueeze(0)

    @staticmethod
    def _cf_boxes_norm(world: "World", index_map: dict, n_cf: int):
        """Boxes of the edited scene: kept objects keep their box; an added object has no
        detected position, so it is placed in the middle of the image with the median
        object size."""
        if world.boxes_norm is None:
            return None
        inverse = {c: f for f, c in index_map.items()}
        sizes = (world.boxes_norm[:, 2:] - world.boxes_norm[:, :2]).median(dim=0).values
        centre = torch.tensor([0.5, 0.5], dtype=world.boxes_norm.dtype, device=world.boxes_norm.device)
        default = torch.cat([centre - sizes / 2, centre + sizes / 2])
        return torch.stack([world.boxes_norm[inverse[c]] if c in inverse else default for c in range(n_cf)])

    @staticmethod
    def _batch(tensor):
        return None if tensor is None else tensor.unsqueeze(0)

    def _decode(self, hidden: torch.Tensor, question_embedding: torch.Tensor) -> torch.Tensor:
        return self.model.decode(hidden.unsqueeze(0), question_embedding)[0]

    def _propagate(self, objects: torch.Tensor, adjacency: torch.Tensor, boxes_norm=None,
                   relation_probs=None) -> torch.Tensor:
        hidden, _ = self.model.run_causal_layers(objects.unsqueeze(0), adjacency.unsqueeze(0),
                                                 self._batch(boxes_norm), self._batch(relation_probs))
        return hidden[0]

    def _encode_one(self, raw: torch.Tensor, class_id: int, box_norm) -> torch.Tensor:
        """Node value of a single (replaced or added) object: DETR embedding + class + box."""
        class_ids = torch.tensor([[class_id]], device=raw.device)
        boxes = None if box_norm is None else box_norm.reshape(1, 1, 4)
        return self.model.encode_nodes(raw.reshape(1, 1, -1), class_ids, boxes)[0, 0]

    @staticmethod
    def _cf_class_ids(world: "World", index_map: dict, n_cf: int, overrides: dict | None = None):
        """Class ids of the edited scene (cf index -> id); ``overrides`` sets replaced/added nodes."""
        if world.class_ids is None:
            return None
        inverse = {c: f for f, c in index_map.items()}
        overrides = overrides or {}
        ids = [overrides.get(c, int(world.class_ids[inverse[c]]) if c in inverse else NO_OBJECT_ID)
               for c in range(n_cf)]
        return torch.tensor(ids, dtype=torch.long, device=world.class_ids.device)

    # ---------------------------------------------------------------
    @torch.no_grad()
    def observe(self, image_path: str, question: str) -> World:
        vision = self.model.vision(image_path)
        raw = vision["embeddings"]
        if raw.dim() == 3:
            raw = raw[0]
        boxes_norm = vision["boxes_norm"][0] if vision.get("boxes_norm") else None
        class_ids = vision["label_ids"][0] if vision.get("label_ids") else None
        world = self.observe_from_tensors(raw, vision["labels"][0], question, boxes_norm, class_ids)
        world.boxes = vision["boxes"][0].detach().cpu().tolist() if vision.get("boxes") else None
        world.scores = vision["scores"][0].detach().cpu().tolist() if vision.get("scores") else None
        return world

    @torch.no_grad()
    def observe_from_tensors(self, raw_objects: torch.Tensor, labels: list[str], question: str,
                             boxes_norm: torch.Tensor | None = None,
                             class_ids: torch.Tensor | None = None) -> World:
        if raw_objects.shape[0] == 0:
            raise RuntimeError("The detector found no objects, so there is no causal graph to intervene on.")
        needs_boxes = (getattr(self.model, "node_identity", False) or getattr(self.model, "relation_messages", False)
                       or getattr(getattr(self.model, "graph", None), "geometry", False))
        if boxes_norm is None and needs_boxes:
            raise RuntimeError("This model needs the object boxes.")
        if class_ids is None:
            class_ids = torch.tensor([label_to_id(label) for label in labels], dtype=torch.long,
                                     device=raw_objects.device)
        q = self.model.text(question)
        objects = self.model.encode_nodes(raw_objects.unsqueeze(0), class_ids.unsqueeze(0), self._batch(boxes_norm))
        graph = self.model.build_graph(objects, q, self._batch(boxes_norm))
        adjacency = graph["adjacency"]
        relation_probs = graph.get("relation_probs")
        hidden, _ = self.model.run_causal_layers(objects, adjacency, self._batch(boxes_norm), relation_probs)
        logits = self.model.decode(hidden, q)
        return World(list(labels), raw_objects, objects[0], adjacency[0], hidden[0], logits[0],
                     question, q, edge_probs=graph["probs"][0], boxes_norm=boxes_norm, class_ids=class_ids,
                     relation_probs=None if relation_probs is None else relation_probs[0])

    # ---------------------------------------------------------------
    def _target_nodes(self, world: World, intervention: Intervention) -> list[int]:
        if intervention.nodes:
            bad = [i for i in intervention.nodes if i < 0 or i >= len(world.labels)]
            if bad:
                raise IndexError(f"Node index {bad} out of range 0..{len(world.labels) - 1}")
            return list(intervention.nodes)
        known = sorted(set(world.labels) | set(self.known_labels))
        label = resolve_label(intervention.target or "", known)
        if label is None:
            raise ValueError(f"Could not map '{intervention.target}' to a detector label.")
        nodes = [i for i, name in enumerate(world.labels) if name.lower() == label.lower()]
        if not nodes:
            raise ValueError(
                f"No '{label}' was detected in the image (detected: {', '.join(world.labels)})."
            )
        return nodes

    def _resolve_new_label(self, phrase: str) -> str:
        if self.steerer is None:
            raise RuntimeError("replace/add need DETR's class head to build x'.")
        label = resolve_label(phrase, self.known_labels)
        if label is None:
            raise ValueError(f"'{phrase}' is not one of the detector's object classes.")
        return label

    # ---------------------------------------------------------------
    @torch.no_grad()
    def intervene(self, world: World, intervention: Intervention) -> CounterfactualResult:
        if intervention.kind == "remove":
            return self._remove(world, intervention)
        if intervention.kind == "replace":
            return self._replace(world, intervention)
        if intervention.kind == "add":
            return self._add(world, intervention)
        if intervention.kind == "attribute":
            return self._attribute(world, intervention)
        raise ValueError(f"Unknown intervention kind: {intervention.kind}")

    def _finish(self, world, intervention, cf_objects, cf_adj, cf_labels, cf_raw,
                intervened, affected_cf, index_map, descendants, steering,
                cf_class_ids=None, cf_relation_probs=None) -> CounterfactualResult:
        if self.mode == "recompute":
            return self._finish_recompute(world, intervention, cf_objects, cf_labels, cf_raw,
                                          intervened, index_map, steering, cf_class_ids)
        # do-mode: recompute only affected nodes; copy factual values for the rest.
        cf_boxes_norm = self._cf_boxes_norm(world, index_map, len(cf_labels))
        full = self._propagate(cf_objects, cf_adj, cf_boxes_norm, cf_relation_probs)
        cf_hidden = full.clone()
        drift = 0.0
        unaffected = []
        for f_idx, c_idx in index_map.items():
            if c_idx in affected_cf:
                continue
            drift = max(drift, float((full[c_idx] - world.hidden[f_idx]).abs().max()))
            cf_hidden[c_idx] = world.hidden[f_idx]       # Eq. 14: non-descendants keep h
            unaffected.append(f_idx)
        cf_logits = self._decode(cf_hidden, world.question_embedding)
        inverse = {c: f for f, c in index_map.items()}
        cf_boxes = None
        if world.boxes is not None:
            cf_boxes = [world.boxes[inverse[c]] if c in inverse else None for c in range(len(cf_labels))]
        cf_world = World(cf_labels, cf_raw, cf_objects, cf_adj, cf_hidden, cf_logits,
                         world.question, world.question_embedding, boxes=cf_boxes, boxes_norm=cf_boxes_norm,
                         class_ids=cf_class_ids, relation_probs=cf_relation_probs)
        return CounterfactualResult(intervention, world, cf_world, intervened, descendants,
                                    sorted(unaffected), drift, steering, index_map)

    def _finish_recompute(self, world, intervention, cf_objects, cf_labels, cf_raw,
                          intervened, index_map, steering, cf_class_ids=None) -> CounterfactualResult:
        """Scene-edit mode: the model re-calculates all relations on the edited scene and
        answers exactly like a normal question:  A' = f_theta(O', q),  h' = SCM layers(O', A'),
        Y' = Decoder(h', q)."""
        q = world.question_embedding
        cf_boxes_norm = self._cf_boxes_norm(world, index_map, len(cf_labels))
        graph = self.model.build_graph(cf_objects.unsqueeze(0), q, self._batch(cf_boxes_norm))
        cf_adj = graph["adjacency"][0]
        cf_probs = graph["probs"][0]
        cf_relation_probs = None if graph.get("relation_probs") is None else graph["relation_probs"][0]
        cf_hidden = self._propagate(cf_objects, cf_adj, cf_boxes_norm, cf_relation_probs)
        cf_logits = self._decode(cf_hidden, q)

        changed, unaffected = [], []
        for f_idx, c_idx in index_map.items():
            if f_idx in intervened:
                continue
            before = world.hidden[f_idx]
            delta = float((cf_hidden[c_idx] - before).detach().norm() / before.norm().clamp(min=1e-8))
            (changed if delta > 1e-4 else unaffected).append(f_idx)

        inverse = {c: f for f, c in index_map.items()}
        cf_boxes = None
        if world.boxes is not None:
            cf_boxes = [world.boxes[inverse[c]] if c in inverse else None for c in range(len(cf_labels))]
        cf_world = World(cf_labels, cf_raw, cf_objects, cf_adj, cf_hidden, cf_logits,
                         world.question, q, boxes=cf_boxes, edge_probs=cf_probs, boxes_norm=cf_boxes_norm,
                         class_ids=cf_class_ids, relation_probs=cf_relation_probs)
        return CounterfactualResult(intervention, world, cf_world, intervened, sorted(changed),
                                    sorted(unaffected), 0.0, steering, index_map, mode="recompute")

    # ---------------------------------------------------------------
    @staticmethod
    def relationship_changes(result: CounterfactualResult, min_change: float = 0.05,
                             top_scores: int = 3) -> dict:
        """How the relations between objects differ between the real and the edited scene.

        Nodes keep their real-scene ids so both graphs can be compared; an added object is
        called "new".  An edge "a -> b" means a influences b (A[b, a] > 0)."""
        factual, cf = result.factual, result.counterfactual
        inverse = {c: f for f, c in result.cf_index_map.items()}
        n_f, n_c = len(factual.labels), len(cf.labels)

        def f_name(f):
            return f"{factual.labels[f]}({f})"

        def c_name(c):
            f = inverse.get(c)
            return f"{cf.labels[c]}({f})" if f is not None else f"{cf.labels[c]}(new)"

        def key_of_c(c):
            f = inverse.get(c)
            return ("f", f) if f is not None else ("n", c)

        before = {}
        for i in range(n_f):
            for j in range(n_f):
                if i != j and factual.adjacency[i, j] > 1e-6:
                    before[(("f", j), ("f", i))] = (float(factual.adjacency[i, j]), f"{f_name(j)} -> {f_name(i)}")
        after = {}
        for i in range(n_c):
            for j in range(n_c):
                if i != j and cf.adjacency[i, j] > 1e-6:
                    after[(key_of_c(j), key_of_c(i))] = (float(cf.adjacency[i, j]), f"{c_name(j)} -> {c_name(i)}")

        removed_ids = {("f", f) for f in result.intervened_nodes
                       if result.intervention.kind == "remove"}
        new_edges = [{"edge": text, "weight": w} for k, (w, text) in after.items() if k not in before]
        intervened_ids = {("f", f) for f in result.intervened_nodes}

        def lost_reason(k):
            if k[0] in removed_ids or k[1] in removed_ids:
                return "object removed"
            if result.mode == "do" and k[1] in intervened_ids:
                return "cut by do()"
            return "relation re-estimated"

        lost_edges = [{"edge": text, "weight": w, "reason": lost_reason(k)}
                      for k, (w, text) in before.items() if k not in after]
        changed_edges = []
        for k in before.keys() & after.keys():
            w0, text0 = before[k]
            w1, text1 = after[k]
            if abs(w1 - w0) >= min_change:
                changed_edges.append({"edge": text1, "before": w0, "after": w1})
        unchanged = sum(1 for k in before.keys() & after.keys()
                        if abs(after[k][0] - before[k][0]) < min_change)

        # links of the changed object(s), before and after (thresholded graph + raw edge scores)
        focus = []
        for node in result.intervened_nodes:
            entry = {}
            if node < n_f:
                entry["before_name"] = f_name(node)
                entry["before_parents"] = [(f_name(j), float(factual.adjacency[node, j]))
                                           for j in range(n_f) if j != node and factual.adjacency[node, j] > 1e-6]
                entry["before_children"] = [(f_name(i), float(factual.adjacency[i, node]))
                                            for i in range(n_f) if i != node and factual.adjacency[i, node] > 1e-6]
                if factual.edge_probs is not None:
                    scores = [(f_name(j), float(factual.edge_probs[node, j])) for j in range(n_f) if j != node]
                    entry["before_top_scores"] = sorted(scores, key=lambda x: -x[1])[:top_scores]
            c = result.cf_index_map.get(node, node if node >= n_f else None)
            if c is not None and c < n_c:
                entry["after_name"] = c_name(c)
                entry["after_parents"] = [(c_name(j), float(cf.adjacency[c, j]))
                                          for j in range(n_c) if j != c and cf.adjacency[c, j] > 1e-6]
                entry["after_children"] = [(c_name(i), float(cf.adjacency[i, c]))
                                           for i in range(n_c) if i != c and cf.adjacency[i, c] > 1e-6]
                if cf.edge_probs is not None:
                    scores = [(c_name(j), float(cf.edge_probs[c, j])) for j in range(n_c) if j != c]
                    entry["after_top_scores"] = sorted(scores, key=lambda x: -x[1])[:top_scores]
            focus.append(entry)

        # objects whose internal state changed most
        state_changes = []
        for f, c in result.cf_index_map.items():
            if f in result.intervened_nodes:
                continue
            delta = float((cf.hidden[c] - factual.hidden[f]).norm() / factual.hidden[f].norm().clamp(min=1e-8))
            if delta > 1e-4:
                state_changes.append({"node": f_name(f), "relative_change": delta})
        state_changes.sort(key=lambda x: -x["relative_change"])

        return {
            "mode": result.mode,
            "new": sorted(new_edges, key=lambda x: -x["weight"]),
            "lost": sorted(lost_edges, key=lambda x: -x["weight"]),
            "changed": sorted(changed_edges, key=lambda x: -abs(x["after"] - x["before"])),
            "unchanged": unchanged,
            "focus": focus,
            "state_changes": state_changes,
        }

    def _remove(self, world: World, intervention: Intervention) -> CounterfactualResult:
        removed = self._target_nodes(world, intervention)
        keep = [i for i in range(len(world.labels)) if i not in removed]
        if not keep:
            raise ValueError("Removing every detected object leaves an empty scene.")
        descendants = [d for d in descendants_of(world.adjacency, removed) if d not in removed]
        index_map = {f: c for c, f in enumerate(keep)}
        keep_t = torch.tensor(keep, device=world.adjacency.device)
        cf_adj = world.adjacency.index_select(0, keep_t).index_select(1, keep_t)
        cf_adj = self._normalise(cf_adj)
        affected_cf = {index_map[d] for d in descendants}
        cf_relation_probs = None
        if world.relation_probs is not None:
            cf_relation_probs = world.relation_probs.index_select(0, keep_t).index_select(1, keep_t)
        return self._finish(world, intervention, world.objects[keep_t], cf_adj,
                            [world.labels[i] for i in keep], world.raw_objects[keep_t],
                            removed, affected_cf, index_map, descendants, {},
                            self._cf_class_ids(world, index_map, len(keep)), cf_relation_probs)

    def _replace(self, world: World, intervention: Intervention) -> CounterfactualResult:
        nodes = self._target_nodes(world, intervention)
        new_label = self._resolve_new_label(intervention.new_label or "")
        cf_raw = world.raw_objects.clone()
        cf_objects = world.objects.clone()
        cf_adj = world.adjacency.clone()
        cf_labels = list(world.labels)
        steering = {}
        new_id = label_to_id(new_label)
        for k in nodes:
            x_prime, prob, ok = self.steerer.steer(world.raw_objects[k], new_label)
            cf_raw[k] = x_prime
            box = None if world.boxes_norm is None else world.boxes_norm[k]
            cf_objects[k] = self._encode_one(x_prime, new_id, box)                         # Eq. 13
            cf_adj[k, :] = 0.0                                                              # Eq. 12
            cf_labels[k] = new_label
            steering[k] = {"label": new_label, "prob": prob, "converged": ok}
        descendants = [d for d in descendants_of(cf_adj, nodes) if d not in nodes]
        index_map = {i: i for i in range(len(world.labels))}
        affected_cf = set(nodes) | set(descendants)
        cf_class_ids = self._cf_class_ids(world, index_map, len(cf_labels), {k: new_id for k in nodes})
        return self._finish(world, intervention, cf_objects, cf_adj, cf_labels, cf_raw,
                            nodes, affected_cf, index_map, descendants, steering,
                            cf_class_ids, world.relation_probs)

    def _attribute(self, world: World, intervention: Intervention) -> CounterfactualResult:
        """do(X_k = x') for an attribute change, e.g. traffic light -> green.

        DETR has no attribute classes, so x' is built in the model's own node
        space with a text-defined direction.  The graph's question projection
        (BERT 768 -> node space 256) is trained jointly with the node values,
        so we use it to embed the attribute:

            d  = P(BERT("a green traffic light")) - P(BERT("a traffic light"))
            x' = h_k + strength * ||h_k|| * d / ||d||

        This is an approximation (the backbone was never trained on attribute
        interventions); the reported similarity shows how far x' moved towards
        the text description.
        """
        attribute = re.sub(r"^(?:a|an|the)\s+", "", (intervention.new_label or "").strip().lower())
        if not attribute:
            raise ValueError("No attribute value given.")
        # "the bicycle became motorcycle" is a class change, not an attribute
        if self.steerer is not None:
            as_class = resolve_label(attribute, self.known_labels)
            if as_class is not None and attribute not in _COLOURS:
                return self._replace(world, Intervention("replace", intervention.target, as_class,
                                                         intervention.nodes))

        if getattr(self.model, "graph", None) is None:
            raise ValueError("Attribute interventions use the relation graph's text projection; "
                             "this model has no graph (--graph-mode none).")
        nodes = self._target_nodes(world, intervention)
        cf_objects = world.objects.clone()
        cf_adj = world.adjacency.clone()
        cf_labels = list(world.labels)
        steering = {}
        for k in nodes:
            name = world.labels[k]
            texts = [f"a {attribute} {name}", f"a {name}"]
            text_nodes = self.model.graph.project_question(self.model.text(texts))   # [2, D]
            direction = text_nodes[0] - text_nodes[1]
            h = world.objects[k]
            x_prime = h + self.attribute_strength * h.norm() * direction / direction.norm().clamp(min=1e-8)
            before = float(F.cosine_similarity(h, text_nodes[0], dim=0))
            after = float(F.cosine_similarity(x_prime, text_nodes[0], dim=0))
            cf_objects[k] = x_prime                                                 # Eq. 13
            cf_adj[k, :] = 0.0                                                      # Eq. 12
            cf_labels[k] = f"{attribute} {name}"
            steering[k] = {"label": cf_labels[k], "prob": after, "converged": after > before,
                           "attribute": True, "similarity_before": before, "similarity_after": after}
        descendants = [d for d in descendants_of(cf_adj, nodes) if d not in nodes]
        index_map = {i: i for i in range(len(world.labels))}
        affected_cf = set(nodes) | set(descendants)
        return self._finish(world, intervention, cf_objects, cf_adj, cf_labels, world.raw_objects.clone(),
                            nodes, affected_cf, index_map, descendants, steering,
                            self._cf_class_ids(world, index_map, len(cf_labels)), world.relation_probs)

    def _add(self, world: World, intervention: Intervention) -> CounterfactualResult:
        new_label = self._resolve_new_label(intervention.new_label or "")
        base = world.raw_objects.mean(dim=0)
        x_prime, prob, ok = self.steerer.steer(base, new_label)
        n = len(world.labels)
        new_id = label_to_id(new_label)
        index_map_add = {i: i for i in range(n)}
        cf_boxes_norm = self._cf_boxes_norm(world, index_map_add, n + 1)
        new_box = None if cf_boxes_norm is None else cf_boxes_norm[n]
        cf_raw = torch.cat([world.raw_objects, x_prime.unsqueeze(0)], dim=0)
        cf_objects = torch.cat([world.objects, self._encode_one(x_prime, new_id, new_box).unsqueeze(0)], dim=0)
        # Learned edge function decides whom the new object influences (its column);
        # its own row stays empty because do() makes it exogenous.
        scored_graph = self.model.build_graph(cf_objects.unsqueeze(0), world.question_embedding,
                                              self._batch(cf_boxes_norm))
        scored = scored_graph["adjacency"][0]
        cf_relation_probs = None
        if scored_graph.get("relation_probs") is not None:
            cf_relation_probs = scored_graph["relation_probs"][0].clone()
            if world.relation_probs is not None:
                cf_relation_probs[:n, :n] = world.relation_probs
        cf_adj = torch.zeros(n + 1, n + 1, dtype=world.adjacency.dtype, device=world.adjacency.device)
        cf_adj[:n, :n] = world.adjacency
        cf_adj[:n, n] = scored[:n, n]
        cf_adj = self._normalise(cf_adj)
        cf_adj[n, :] = 0.0
        descendants_cf = descendants_of(cf_adj, [n])
        descendants = [d for d in descendants_cf if d < n]
        index_map = {i: i for i in range(n)}
        affected_cf = set(descendants) | {n}
        return self._finish(world, intervention, cf_objects, cf_adj,
                            world.labels + [new_label], cf_raw, [n], affected_cf, index_map,
                            descendants, {n: {"label": new_label, "prob": prob, "converged": ok}},
                            self._cf_class_ids(world, index_map, n + 1, {n: new_id}), cf_relation_probs)

    # ---------------------------------------------------------------
    def resolver(self, world: World):
        """phrase -> detector label, using the scene's labels and every DETR class."""
        known = sorted(set(world.labels) | set(self.known_labels))
        return lambda phrase: resolve_label(phrase, known)

    @staticmethod
    def scene_objects(world: World, result: CounterfactualResult | None = None) -> list[SceneObject]:
        """Scene graph nodes for the scene-graph answerer.  With ``result`` the
        counterfactual world is returned: replaced nodes carry their new class,
        attribute interventions become node attributes, added nodes have no box."""
        if result is None:
            boxes = world.boxes or [None] * len(world.labels)
            return [SceneObject(i, label, tuple(box) if box is not None else None)
                    for i, (label, box) in enumerate(zip(world.labels, boxes))]
        cf = result.counterfactual
        inverse = {c: f for f, c in result.cf_index_map.items()}
        boxes = cf.boxes or [None] * len(cf.labels)
        objects = []
        for c, box in enumerate(boxes):
            f = inverse.get(c)
            info = result.steering.get(f if f is not None else c, {})
            label = cf.labels[c]
            attributes: list[str] = []
            if info.get("attribute") and f is not None:
                label = result.factual.labels[f]
                attributes = [cf.labels[c][: -len(label)].strip()]
            objects.append(SceneObject(c, label, tuple(box) if box is not None else None, attributes))
        return objects

    @staticmethod
    def _node_name(result: CounterfactualResult, i: int) -> str:
        factual = result.factual.labels
        if i >= len(factual):                       # node created by add
            return f"{i}:{result.counterfactual.labels[i]} (new)"
        if result.intervention.kind in {"replace", "attribute"}:
            return f"{i}:{factual[i]} -> {result.counterfactual.labels[i]}"
        return f"{i}:{factual[i]}"

    @staticmethod
    def effect_report(result: CounterfactualResult) -> dict:
        """What changes under do(): the scene change, which objects depend on the
        intervened node(s) in the learned causal graph, and how much each of
        their representations moves.  Used when the user asks a bare
        "what if ...?" without a follow-up question."""
        from collections import Counter

        from models.scene_answer import _plural

        factual, cf = result.factual, result.counterfactual
        kind = result.intervention.kind
        nodes = result.intervened_nodes

        def group(indices, labels):
            counts = Counter(labels[i] for i in indices)
            return ", ".join(f"{n} {_plural(label, n)}" for label, n in counts.items())

        def listing(labels):
            counts = Counter(labels)
            return ", ".join(f"{n} {_plural(label, n)}" for label, n in counts.most_common()) or "nothing"

        node_text = ", ".join(str(i) for i in nodes)
        if kind == "remove":
            change = f"removed {group(nodes, factual.labels)} (node {node_text})"
        elif kind == "add":
            change = f"added a {cf.labels[nodes[0]]} (new node {node_text})"
        else:
            change = "; ".join(f"{factual.labels[i]} (node {i}) -> {cf.labels[i]}" for i in nodes)

        # edges cut by do(): parents of the intervened nodes
        cut = []
        if kind in {"replace", "attribute", "remove"} and result.mode == "do":
            for i in nodes:
                for j in range(len(factual.labels)):
                    if j != i and factual.adjacency[i, j] > 1e-6 and j not in nodes:
                        cut.append(f"{factual.labels[j]} ({j}) -> {factual.labels[i]} ({i})")

        # how strongly each descendant changed (relative change of its node state)
        index_map = result.cf_index_map
        graph = factual.adjacency if kind != "add" else cf.adjacency
        sources = nodes
        effects = []
        for d in result.descendants:
            c = index_map.get(d, d)
            before = factual.hidden[d]
            after = cf.hidden[c]
            change_size = float((after - before).norm() / before.norm().clamp(min=1e-8))
            direct = any(graph[d if kind != "add" else c, src] > 1e-6 for src in sources if src < graph.shape[1])
            effects.append({"node": d, "label": factual.labels[d], "relative_change": change_size,
                            "direct": bool(direct)})
        effects.sort(key=lambda item: -item["relative_change"])

        labels_of_target = factual.labels if kind != "add" else cf.labels
        base = labels_of_target[nodes[0]]
        plural = len(nodes) > 1
        target = _plural(base, len(nodes))
        verb, self_word = ("depend", "themselves") if plural else ("depends", "itself")
        if result.mode == "recompute":
            if effects:
                causes = ", ".join(f"{e['label']} (node {e['node']}, changes {100 * e['relative_change']:.0f}%)"
                                   for e in effects)
                conclusion = (f"After the model re-calculates the relations in the edited scene, the "
                              f"objects whose state changes are: {causes}.")
            else:
                conclusion = ("After the model re-calculates the relations in the edited scene, no other "
                              f"object's state changes; only the {target} {self_word} "
                              f"{'change' if plural else 'changes'}.")
        elif effects:
            causes = ", ".join(
                f"{e['label']} (node {e['node']}, {'direct' if e['direct'] else 'indirect'}, "
                f"changes {100 * e['relative_change']:.0f}%)" for e in effects
            )
            conclusion = (f"In the learned causal graph the {target} {'influence' if plural else 'influences'}: "
                          f"{causes}. All other objects keep their original state.")
        else:
            conclusion = (f"In the learned causal graph no other object {verb.rstrip('s') + 's'} on the {target}, "
                          f"so nothing else in the scene changes; only the {target} {self_word} "
                          f"{'change' if plural else 'changes'}.")

        return {
            "target_label": base,
            "scene_change": change,
            "scene_before": listing(factual.labels),
            "scene_after": listing(cf.labels),
            "edges_cut": cut,
            "affected_objects": effects,
            "conclusion": conclusion,
        }

    def summarise(self, result: CounterfactualResult, k: int = 5) -> dict:
        f_probs = F.softmax(result.factual.logits, dim=-1)
        c_probs = F.softmax(result.counterfactual.logits, dim=-1)
        f_idx = int(f_probs.argmax())
        c_idx = int(c_probs.argmax())
        return {
            "intervention": result.intervention.describe(result.mode),
            "factual_answer": self.answer_name(f_idx),
            "factual_confidence": float(f_probs[f_idx]),
            "counterfactual_answer": self.answer_name(c_idx),
            "counterfactual_confidence": float(c_probs[c_idx]),
            "answer_changed": f_idx != c_idx,
            "factual_top": self.top_answers(result.factual.logits, k),
            "counterfactual_top": self.top_answers(result.counterfactual.logits, k),
            # causal effect of the intervention on the factual answer
            "effect_on_factual_answer": float(c_probs[f_idx] - f_probs[f_idx]),
            "total_variation": float(0.5 * (c_probs - f_probs).abs().sum()),
            "intervened_nodes": [self._node_name(result, i) for i in result.intervened_nodes],
            "descendants": [f"{i}:{result.factual.labels[i]}" for i in result.descendants],
            "unaffected_nodes": len(result.unaffected),
            "non_descendant_max_drift": result.non_descendant_max_drift,
            "steering": result.steering,
        }
