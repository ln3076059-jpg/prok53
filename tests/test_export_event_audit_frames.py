from training.dmd.export_event_audit_frames import _sample_times


def test_sample_times_prioritizes_high_confidence_candidate_and_stays_in_interval() -> None:
    rows = [
        {"timestamp": 3.0, "confidence": 0.2},
        {"timestamp": 7.0, "confidence": 0.9},
    ]

    timestamps = _sample_times(2.0, 8.0, rows, limit=4)

    assert 7.0 in timestamps
    assert 3.0 in timestamps
    assert all(2.0 <= value <= 8.0 for value in timestamps)
    assert timestamps == sorted(timestamps)


def test_sample_times_deduplicates_nearby_candidates() -> None:
    rows = [
        {"timestamp": 1.00, "confidence": 0.9},
        {"timestamp": 1.02, "confidence": 0.8},
    ]

    timestamps = _sample_times(1.0, 1.2, rows, limit=4)

    assert sum(abs(value - 1.0) <= 0.05 for value in timestamps) == 1
