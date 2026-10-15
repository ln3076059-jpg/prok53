"""Governed integrity tests for Roadwatch V7 research results and artifacts."""

import json
from pathlib import Path

V7_RESULT_PATH = Path("reports/v7_optimization/v7_experiments.json")


def _v7_result():
    return json.loads(V7_RESULT_PATH.read_text(encoding="utf-8"))


def test_v7_loso_is_subject_disjoint_and_does_not_access_holdouts():
    result = _v7_result()
    development = set(result["development_subjects"])

    assert result["holdouts_accessed"] == []
    assert set(result["forbidden_consumed_holdouts"]) == {"gZ-37", "gE-28"}
    assert development == {"gB-9", "gZ-36", "gC-14"}

    for variant in result["variants"]:
        for fold in variant["folds"]:
            train = set(fold["training_subjects"])
            heldout = fold["heldout_subject"]
            assert heldout not in train
            assert train | {heldout} == development


def test_v7_pooled_event_counts_and_metrics_reconcile():
    result = _v7_result()
    for variant in result["variants"]:
        for experiment in variant["experiments"]:
            metrics = experiment["pooled_metrics"]
            tp = metrics["true_positives"]
            fp = metrics["false_positives"]
            fn = metrics["false_negatives"]
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )

            assert abs(metrics["precision"] - precision) <= 0.0005
            assert abs(metrics["recall"] - recall) <= 0.0005
            assert abs(metrics["f1"] - f1) <= 0.0005
            assert tp + fn == 29


def test_v7_clip_confusion_matrices_reconcile():
    result = _v7_result()
    for variant in result["variants"]:
        metrics = variant["clip_metrics"]
        matrix_total = sum(sum(row) for row in metrics["confusion_matrix"])
        support_total = sum(row["support"] for row in metrics["per_class"].values())
        assert matrix_total == metrics["evaluated_clips"] == support_total


def test_v7_selection_gate_and_governance():
    result = _v7_result()
    # Check that no candidate passed all gates
    assert result["selected_candidate"] is None
    assert all(not row["eligible"] for row in result["selection_rows"])
    assert result["governance"]["production_approved"] is False
    assert result["governance"]["automatic_violation"] is False
    assert result["governance"]["activation_state"] == "DEVELOPMENT_ONLY_NOT_VALIDATED"


def test_v7_checkpoints_and_predictions_exist():
    result = _v7_result()
    for variant in result["variants"]:
        if variant["variant"]["id"] in {
            "V7_A_V6_BASELINE",
            "V7_F_NEGATIVE_MARGIN",
        }:
            continue
        for fold in variant["folds"]:
            checkpoint = Path(fold["checkpoint_path"])
            prediction = Path(fold["prediction_path"])
            assert checkpoint.is_file(), f"missing checkpoint: {checkpoint}"
            assert prediction.is_file(), f"missing prediction: {prediction}"
