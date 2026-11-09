"""Scene-graph answers in factual vs. counterfactual scenes (no model needed)."""

from models.do_intervention import resolve_label
from models.scene_answer import SceneObject, answer_from_scene

LABELS = ["person", "bicycle", "motorcycle", "umbrella", "car", "bus", "couch", "chair",
          "traffic light", "dog", "cat", "skateboard"]


def resolve(phrase):
    return resolve_label(phrase, LABELS)


def street():
    return [
        SceneObject(0, "person", (400, 300, 520, 600)),
        SceneObject(1, "bicycle", (380, 450, 560, 650)),
        SceneObject(2, "umbrella", (390, 200, 540, 320)),
        SceneObject(3, "car", (0, 400, 200, 520)),
        SceneObject(4, "car", (650, 380, 820, 500)),
        SceneObject(5, "bus", (850, 250, 1100, 500)),
        SceneObject(6, "person", (900, 350, 950, 480)),       # passenger inside the bus
        SceneObject(7, "traffic light", (700, 50, 740, 150)),
    ]


def test_relation_prefers_rideable_object_over_bus():
    assert answer_from_scene("What is the man riding?", street(), resolve).answer == "bicycle"


def test_relation_after_replace():
    scene = street()
    scene[1].label = "motorcycle"
    assert answer_from_scene("what is the man riding?", scene, resolve).answer == "motorcycle"


def test_holding_needs_contact_and_plausible_object():
    assert answer_from_scene("What is the person holding?", street(), resolve).answer == "umbrella"
    without_umbrella = [obj for obj in street() if obj.label != "umbrella"]
    assert answer_from_scene("What is the person holding?", without_umbrella, resolve).answer == "nothing"


def test_counting_and_existence():
    assert answer_from_scene("How many cars are there?", street(), resolve).answer == "2"
    assert answer_from_scene("how many people", street(), resolve).answer == "2"
    assert answer_from_scene("Is there a dog?", street(), resolve).answer == "no"
    assert answer_from_scene("Are there any buses?", street(), resolve).answer == "yes"


def test_colour_from_attribute_intervention():
    scene = street()
    assert answer_from_scene("What color is the traffic light?", scene, resolve).answer == "unknown"
    scene[7].attributes = ["green"]
    assert answer_from_scene("What color is the traffic light?", scene, resolve).answer == "green"
    assert answer_from_scene("Is the light red?", scene, resolve).answer == "no"


def test_added_object_without_box():
    scene = [SceneObject(0, "person", (100, 100, 200, 400)), SceneObject(1, "skateboard")]
    assert answer_from_scene("What is the person riding?", scene, resolve).answer == "skateboard"


def test_unsupported_question_returns_none():
    assert answer_from_scene("Why is the street wet?", street(), resolve) is None
