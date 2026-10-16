import numpy as np
import pytest

from training.dmd.ai85_next import (
    EventPolicy,
    aggregate_event_metrics,
    interval_sample_times,
    interval_trace,
    longest_true_run,
    normalize_per_video,
    policy_rank,
    prediction_centers,
)


def test_per_video_normalization_removes_offset_without_nonfinite_values() -> None:
    values = np.asarray([[5.0, 2.0, 10.0], [7.0, 2.0, 14.0], [9.0, 2.0, 18.0]])
    normalized, diagnostics = normalize_per_video(values)

    assert np.allclose(normalized.mean(axis=0), 0.0, atol=1e-6)
    assert np.allclose(normalized[:, 1], 0.0)
    assert np.isfinite(normalized).all()
    assert diagnostics["constant_or_near_constant_features"] == 1


def test_prediction_centers_match_nine_sample_legacy_coverage() -> None:
    assert prediction_centers(12).tolist() == [4, 5, 6, 7]
    assert prediction_centers(8).size == 0


def test_interval_trace_reports_missing_and_continuous_activation() -> None:
    timestamps = np.asarray([1.0, 1.5, 2.0, 2.5])
    probabilities = np.asarray([0.2, 0.9, 0.85, 0.1])

    trace = interval_trace(timestamps, probabilities, 1.4, 2.1, threshold=0.8)
    missing = interval_trace(timestamps, probabilities, 4.0, 5.0, threshold=0.8)

    assert trace["samples"] == 2
    assert trace["longest_activation_run"] == 2
    assert trace["fraction_at_or_above_threshold"] == 1.0
    assert missing["missing_centered_samples"] is True


def test_longest_true_run_handles_fragmented_evidence() -> None:
    assert longest_true_run(np.asarray([True, True, False, True])) == 2
    assert longest_true_run(np.asarray([], dtype=bool)) == 0


def test_policy_rank_enforces_false_alarm_gate_before_f1() -> None:
    safe = {"false_alarms_per_minute": 0.1, "f1": 0.5, "recall": 0.5, "precision": 0.5}
    noisy = {"false_alarms_per_minute": 0.2, "f1": 0.9, "recall": 0.9, "precision": 0.9}
    policy = EventPolicy(0.8, 2, 2.0)

    assert policy_rank(safe, policy) < policy_rank(noisy, policy)


def test_event_aggregation_rejects_subject_set_mismatch() -> None:
    with pytest.raises(ValueError, match="subject sets"):
        aggregate_event_metrics({"a": []}, {"b": []}, {"a": 10.0})


def test_interval_sampling_covers_short_event_without_crossing_boundaries() -> None:
    times = interval_sample_times(10.0, 10.4, 5)

    assert len(times) == 5
    assert np.all(times >= 10.0)
    assert np.all(times <= 10.4)
    assert interval_sample_times(2.0, 2.0, 5).tolist() == [2.0]
