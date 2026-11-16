import json
import math

import pytest

from claimsight.models.damage_data import read_annotation, split_of
from evals.detector_eval import GOLDEN, average_precision, detection_map, iou

T = [{"file": "a.jpg", "boxes": [["dent", 0, 0, 10, 10], ["scratch", 20, 20, 30, 40]]}]


def test_iou():
    assert iou([0, 0, 10, 10], [0, 0, 10, 10]) == 1.0
    assert iou([0, 0, 10, 10], [5, 0, 15, 10]) == pytest.approx(1 / 3)
    assert iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0


def test_perfect_predictions_score_one():
    preds = [{"file": "a.jpg", "boxes": [[b[0], 0.9, *b[1:]] for b in T[0]["boxes"]]}]
    m = detection_map(preds, T)
    assert m["map"] == m["map_50"] == 1.0


def test_wrong_class_and_missing_image_score_zero():
    wrong = [{"file": "a.jpg", "boxes": [["scratch", 0.9, 0, 0, 10, 10]]}]
    assert detection_map(wrong, T)["per_class"]["dent"]["ap"] == 0.0
    assert detection_map([], T)["map"] == 0.0


def test_loose_box_counts_at_iou_50_not_at_75():
    # IoU 0.64: a hit at thresholds 0.50-0.60 (3 of 10), a miss above.
    preds = [{"file": "a.jpg", "boxes": [["dent", 0.9, 0, 0, 10, 6.4]]}]
    c = detection_map(preds, T)["per_class"]["dent"]
    assert (c["ap50"], c["ap75"]) == (1.0, 0.0)
    assert c["ap"] == pytest.approx(0.3)


def test_a_false_positive_ranked_first_halves_precision():
    # One true box; detections: FP (0.9) then TP (0.8) -> precision 0.5 at every recall.
    ap = average_precision([(0.9, False), (0.8, True)], 1)
    assert ap == pytest.approx(0.5)
    assert math.isnan(average_precision([], 0))


def test_split_is_stable_and_roughly_80_10_10():
    names = [f"Car damages {i}.jpg" for i in range(2000)]
    counts = {s: sum(split_of(n) == s for n in names) for s in ("train", "val", "test")}
    assert split_of("x.jpg") == split_of("x.jpg")
    assert 1500 < counts["train"] < 1700 and 150 < counts["test"] < 250


def test_polygons_become_boxes_and_parts_are_dropped(tmp_path):
    ann = {
        "size": {"width": 100, "height": 50},
        "objects": [
            {"classTitle": "Dent", "points": {"exterior": [[10, 5], [30, 8], [20, 25]]}},
            {"classTitle": "Hood", "points": {"exterior": [[0, 0], [90, 40]]}},
        ],
    }
    path = tmp_path / "Car damages 1.jpg.json"
    path.write_text(json.dumps(ann))
    rec = read_annotation(path)
    assert rec["file"] == "Car damages 1.jpg"
    assert rec["boxes"] == [("dent", 10.0, 5.0, 30.0, 25.0)]


@pytest.mark.skipif(
    not (GOLDEN / "cassettes" / "detector.json").exists(), reason="no detector cassette"
)
def test_cassette_covers_every_test_image():
    from evals.detector_eval import load

    preds, truth, _ = load()
    assert {p["file"] for p in preds} == {t["file"] for t in truth}


def test_detector_labels_map_onto_pipeline_damage_types():
    from claimsight.models.detector import pipeline_damage_type

    assert pipeline_damage_type("broken_part", "headlamp") == "lamp_broken"
    assert pipeline_damage_type("broken_part", "windshield") == "glass_shatter"
    assert pipeline_damage_type("broken_part", "door") == "broken_part"
    assert pipeline_damage_type("paint_chip", "hood") == "scratch"
    assert pipeline_damage_type("crack", "windshield") == "crack"
