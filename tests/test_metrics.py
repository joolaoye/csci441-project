"""Accuracy and macro-F1 on a hand-computed confusion matrix."""

import pytest

from traffic_signs.data.labels import CLASS_NAMES, class_name_for_model
from traffic_signs.evaluation.metrics import evaluate_predictions, write_predictions_csv


def test_partial_predictions_match_hand_computed_scores():
    result = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1], num_classes=2)
    assert result["accuracy"] == pytest.approx(0.75)
    assert result["confusion_matrix"] == [[1, 1], [0, 2]]
    assert result["per_class"][0]["precision"] == pytest.approx(1.0)
    assert result["per_class"][0]["recall"] == pytest.approx(0.5)
    assert result["per_class"][0]["f1"] == pytest.approx(2 / 3)
    assert result["per_class"][0]["support"] == 2
    assert result["per_class"][1]["precision"] == pytest.approx(2 / 3)
    assert result["per_class"][1]["recall"] == pytest.approx(1.0)
    assert result["per_class"][1]["f1"] == pytest.approx(0.8)
    assert result["macro_precision"] == pytest.approx((1.0 + 2 / 3) / 2)
    assert result["macro_recall"] == pytest.approx((0.5 + 1.0) / 2)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.8) / 2)


def test_macro_f1_includes_classes_with_no_support():
    result = evaluate_predictions([0, 0, 1, 1], [0, 1, 1, 1], num_classes=3)
    assert result["confusion_matrix"] == [[1, 1, 0], [0, 2, 0], [0, 0, 0]]
    assert result["per_class"][2]["precision"] == 0
    assert result["per_class"][2]["recall"] == 0
    assert result["per_class"][2]["f1"] == 0
    assert result["per_class"][2]["support"] == 0
    assert result["macro_precision"] == pytest.approx((1.0 + 2 / 3 + 0.0) / 3)
    assert result["macro_recall"] == pytest.approx((0.5 + 1.0 + 0.0) / 3)
    assert result["macro_f1"] == pytest.approx((2 / 3 + 0.8 + 0.0) / 3)
    assert result["accuracy"] == pytest.approx(0.75)


def test_perfect_predictions_score_one():
    result = evaluate_predictions([0, 1, 2, 2], [0, 1, 2, 2], num_classes=3)
    assert result["accuracy"] == 1
    assert result["macro_precision"] == 1
    assert result["macro_recall"] == 1
    assert result["macro_f1"] == 1


def test_predictions_csv_records_ids_and_predicted_confidence(tmp_path):
    path = tmp_path / "predictions.csv"
    write_predictions_csv(
        path,
        [{"id": "DATA/0/a.png", "label": 1}, {"id": "DATA/0/b.png", "label": 0}],
        [[0.1, 0.9], [0.7, 0.3]],
    )
    rows = path.read_text(encoding="utf-8").splitlines()
    assert rows[0] == "id,true_class,predicted_class,confidence"
    assert rows[1] == "DATA/0/a.png,1,1,0.900000"
    assert rows[2] == "DATA/0/b.png,0,0,0.700000"


def test_gtsrb_class_names_match_the_official_ids():
    assert len(CLASS_NAMES) == 43
    assert class_name_for_model(0, 43) == "Speed limit (20km/h)"
    assert class_name_for_model(14, 43) == "Stop"
    assert class_name_for_model(17, 43) == "No entry"
    assert class_name_for_model(42, 43) == "End of no passing by vehicles over 3.5 metric tons"
    assert class_name_for_model(14, 2) is None
