from __future__ import annotations

import json
from pathlib import Path

from training.dmd.analyze_event_evidence import (
    _deduplicated_matches,
    _duration_bucket,
    _event_boundary_summary,
    _numeric_summary,
    analyze_subject,
)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text("\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8")


def test_duration_buckets_use_non_overlapping_boundaries() -> None:
    assert _duration_bucket(0.99) == "lt_1s"
    assert _duration_bucket(1.0) == "1_to_3s"
    assert _duration_bucket(3.0) == "3_to_10s"
    assert _duration_bucket(10.0) == "gte_10s"


def test_numeric_summary_handles_empty_and_populated_values() -> None:
    assert _numeric_summary([])["median"] is None
    summary = _numeric_summary([1.0, 2.0, 3.0])
    assert summary["count"] == 3
    assert summary["median"] == 2.0
    assert summary["mean"] == 2.0


def test_event_boundary_summary_uses_only_matched_events() -> None:
    summary = _event_boundary_summary([
        {
            "matched": True,
            "match": {
                "iou": 0.5,
                "start_latency_seconds": -0.25,
                "end_latency_seconds": 0.5,
                "duration_ratio": 0.75,
            },
        },
        {"matched": False},
    ])
    assert summary["temporal_iou"]["count"] == 1
    assert summary["absolute_start_error_seconds"]["mean"] == 0.25
    assert summary["absolute_end_error_seconds"]["mean"] == 0.5


def test_duplicate_gt_matches_keep_best_iou_and_become_false_positive() -> None:
    ground_truth = {
        "start_seconds": 1.0,
        "end_seconds": 5.0,
        "source_action": "driver_actions/phonecall_right",
    }
    metrics = {
        "matched_events": [
            {"ground_truth": ground_truth, "prediction": {"id": "weak"}, "iou": 0.2},
            {"ground_truth": ground_truth, "prediction": {"id": "best"}, "iou": 0.8},
        ]
    }

    best, duplicates = _deduplicated_matches(metrics)

    assert [row["prediction"]["id"] for row in best] == ["best"]
    assert [row["prediction"]["id"] for row in duplicates] == ["weak"]


def test_analyze_subject_connects_selected_candidate_to_source(tmp_path: Path) -> None:
    ground_truth = {
        "start_seconds": 1.0,
        "end_seconds": 2.0,
        "source_action": "driver_actions/texting_right",
    }
    prediction = {
        "start_seconds": 1.1,
        "end_seconds": 1.9,
        "confidence": 0.8,
    }
    metrics = {
        "true_positives": 1,
        "false_positives": 0,
        "false_negatives": 0,
        "total_gt_events": 1,
        "precision": 1.0,
        "recall": 1.0,
        "f1": 1.0,
        "false_alarms_per_minute": 0.0,
        "effective_fps": 2.0,
        "wall_time_seconds": 1.0,
        "matched_events": [{
            "ground_truth": ground_truth,
            "prediction": prediction,
            "iou": 0.8,
            "start_latency": 0.1,
            "end_latency": -0.1,
        }],
        "unmatched_gt": [],
        "unmatched_predictions": [],
    }
    (tmp_path / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    _write_jsonl(tmp_path / "autonomous_predictions.jsonl", [{
        "timestamp": 1.5,
        "candidate_id": "45:0:specialist_roi",
        "confidence": 0.8,
        "rejection_reason": "ADMITTED",
    }])
    _write_jsonl(tmp_path / "phone_candidate_records.jsonl", [{
        "timestamp": 1.5,
        "candidate_id": "45:0:specialist_roi",
        "confidence": 0.8,
        "source": "specialist_roi",
        "assoc_state": "DRIVER_ASSOCIATED",
        "box": [10.0, 20.0, 20.0, 40.0],
    }])

    result = analyze_subject(tmp_path)

    assert result["evidence_totals"]["selected_sources"] == {"specialist_roi": 1}
    event = result["ground_truth_events"][0]
    assert event["matched"] is True
    assert event["proposal_sources"] == {"specialist_roi": 1}
    assert event["admitted_count"] == 1
    assert event["match"]["duration_ratio"] == 0.8
