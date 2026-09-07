"""
Unit tests for evaluation metrics (classification, text, visual grounding, change detection).
"""

import pytest
from eval.metrics import Evaluator


def test_iou_identical_boxes():
    box = [10.0, 10.0, 50.0, 50.0]
    iou = Evaluator.calculate_iou(box, box)
    assert pytest.approx(iou, 1e-5) == 1.0


def test_iou_non_overlapping():
    box1 = [0.0, 0.0, 10.0, 10.0]
    box2 = [20.0, 20.0, 30.0, 30.0]
    iou = Evaluator.calculate_iou(box1, box2)
    assert iou == 0.0


def test_iou_partial_overlap():
    box1 = [0.0, 0.0, 10.0, 10.0]  # area 100
    box2 = [5.0, 0.0, 15.0, 10.0]  # area 100, intersection [5,0,10,10] area 50
    # union = 100 + 100 - 50 = 150 -> iou = 50 / 150 = 1/3
    iou = Evaluator.calculate_iou(box1, box2)
    assert pytest.approx(iou, 1e-3) == 1.0 / 3.0


def test_grounding_metrics():
    ev = Evaluator()
    pred_boxes = [[0.0, 0.0, 10.0, 10.0], [0.0, 0.0, 10.0, 10.0]]
    ref_boxes  = [[0.0, 0.0, 10.0, 10.0], [50.0, 50.0, 60.0, 60.0]]
    metrics = ev.calculate_grounding_metrics(pred_boxes, ref_boxes)
    assert "mean_iou" in metrics
    assert "acc_at_05" in metrics
    assert metrics["acc_at_05"] == 0.5  # 1 match, 1 mismatch


def test_classification_metrics():
    ev = Evaluator()
    preds = ["Yes", "No", "Yes", "Yes"]
    refs  = ["Yes", "No", "No",  "Yes"]
    res = ev.calculate_classification_metrics(preds, refs)
    assert res["accuracy"] == 0.75
    assert 0.0 <= res["f1_macro"] <= 1.0
