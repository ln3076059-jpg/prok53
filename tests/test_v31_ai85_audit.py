from __future__ import annotations

from training.dmd.audit_v31_ai85_plan import (
    _future_dependent_fusion_probe,
    _optimization_sanity_probe,
    _temporal_metrics,
)
from training.dmd.temporal_eval import TemporalMatchingPolicy


def test_v31_temporal_metrics_charges_duplicate_prediction_as_fp() -> None:
    truth = [{"start_seconds": 1.0, "end_seconds": 4.0}]
    predictions = [
        {"start_seconds": 1.0, "end_seconds": 2.0},
        {"start_seconds": 1.2, "end_seconds": 2.2},
    ]
    metrics = _temporal_metrics(
        truth,
        predictions,
        duration_seconds=60.0,
        policy=TemporalMatchingPolicy(),
    )
    assert metrics["true_positives"] == 1
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 0


def test_v31_probe_detects_future_dependent_interval_fusion() -> None:
    probe = _future_dependent_fusion_probe()
    assert probe["future_influence_detected"] is True
    assert probe["label_with_left_future"] != probe["label_with_right_future"]


def test_v31_training_wiring_can_overfit_one_minibatch() -> None:
    probe = _optimization_sanity_probe()
    assert probe["finite_gradients"] is True
    assert probe["pass"] is True
