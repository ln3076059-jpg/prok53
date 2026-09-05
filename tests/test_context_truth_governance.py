import csv
import json
from pathlib import Path

import pytest

from training.evaluate_events import evaluate
from training.freeze_context_ground_truth import (
    REQUIRED_COLUMNS,
    freeze_context_ground_truth,
)


def _write_context(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerows(rows)


def _context_row(**overrides: str) -> dict[str, str]:
    row = {
        "video_id": "video-1",
        "context_id": "context-1",
        "occupant_id": "occupant-1",
        "vehicle_id": "vehicle-1",
        "cabin_id": "cabin-1",
        "occupant_role": "driver",
        "start_seconds": "0",
        "end_seconds": "60",
        "timeline_end_seconds": "60",
        "inside_vehicle": "true",
        "outside_vehicle_person": "false",
        "motorcycle_flag": "false",
        "phone_state": "NO_PHONE",
        "seatbelt_state": "FASTENED",
        "visibility": "clear",
        "conditions": "daylight",
        "human_review_status": "APPROVED",
        "reviewer_id": "reviewer-1",
        "reviewer_type": "HUMAN",
        "reviewed_at": "2026-09-05T00:00:00Z",
        "adjudication_status": "FINAL",
        "notes": "reviewed timeline context",
    }
    row.update(overrides)
    return row


def _external_lock(path: Path) -> None:
    path.write_text(
        json.dumps(
            {
                "status": "FROZEN_EXTERNAL_TEST",
                "human_review_status": "ALL_APPROVED",
                "video_ids": ["video-1"],
            }
        ),
        encoding="utf-8",
    )


def _prediction(**overrides: str) -> dict[str, str]:
    row = {
        "video_id": "video-1",
        "event_type": "PHONE",
        "occupant_id": "occupant-1",
        "occupant_role": "driver",
        "vehicle_id": "vehicle-1",
        "cabin_id": "cabin-1",
        "start_seconds": "30",
        "end_seconds": "31",
        "label": "PHONE_USE",
        "visibility": "clear",
        "outside_vehicle_person": "false",
        "motorcycle_flag": "false",
        "observation_count": "5",
    }
    row.update(overrides)
    return row


def _event_truth() -> list[dict[str, str]]:
    return [
        {
            "video_id": "video-1",
            "event_type": "PHONE",
            "occupant_id": "occupant-1",
            "occupant_role": "driver",
            "vehicle_id": "vehicle-1",
            "cabin_id": "cabin-1",
            "start_seconds": "10",
            "end_seconds": "15",
            "inside_vehicle": "true",
            "outside_vehicle_person": "false",
            "motorcycle_flag": "false",
            "label": "PHONE_USE",
        }
    ]


def test_context_freeze_requires_gapless_full_timeline(tmp_path: Path) -> None:
    external_lock = tmp_path / "external-lock.json"
    _external_lock(external_lock)
    context_path = tmp_path / "context.csv"
    _write_context(
        context_path,
        [
            _context_row(end_seconds="20"),
            _context_row(context_id="context-2", start_seconds="21", end_seconds="60"),
        ],
    )

    with pytest.raises(ValueError, match="gap before context-2"):
        freeze_context_ground_truth(context_path, external_lock, tmp_path / "context-lock.json")


def test_context_freeze_records_identity_coverage_and_duration(tmp_path: Path) -> None:
    external_lock = tmp_path / "external-lock.json"
    _external_lock(external_lock)
    context_path = tmp_path / "context.csv"
    _write_context(
        context_path,
        [
            _context_row(end_seconds="20"),
            _context_row(context_id="context-2", start_seconds="20", end_seconds="60"),
        ],
    )

    frozen = freeze_context_ground_truth(
        context_path, external_lock, tmp_path / "context-lock.json"
    )

    assert frozen["coverage_status"] == "FULL_TIMELINE_NO_GAPS_OR_OVERLAPS"
    assert frozen["identity_status"] == "KNOWN_STABLE_OCCUPANT_VEHICLE_CABIN"
    assert frozen["total_timeline_seconds"] == 60.0


def test_context_freeze_rejects_unknown_identity(tmp_path: Path) -> None:
    external_lock = tmp_path / "external-lock.json"
    _external_lock(external_lock)
    context_path = tmp_path / "context.csv"
    _write_context(context_path, [_context_row(cabin_id="UNKNOWN")])

    with pytest.raises(ValueError, match="cabin_id must be a known stable identity"):
        freeze_context_ground_truth(context_path, external_lock, tmp_path / "context-lock.json")


def test_background_false_positive_uses_separate_context_truth() -> None:
    prediction = _prediction()
    report = evaluate(
        truth_rows=_event_truth(),
        prediction_rows=[prediction],
        video_minutes=1.0,
        prediction_fieldnames=set(prediction),
        context_rows=[_context_row()],
    )

    assert report["event_types"]["PHONE"]["false_positives"] == 1
    assert report["event_types"]["PHONE"]["missed_events"] == 1
    assert report["safety_invariant_counters"] != "NOT_EVALUABLE"


def test_safety_context_requires_exact_occupant_identity() -> None:
    prediction = _prediction(occupant_id="occupant-B", occupant_role="driver")
    context = _context_row(occupant_id="occupant-A", occupant_role="front_passenger")

    report = evaluate(
        truth_rows=_event_truth(),
        prediction_rows=[prediction],
        video_minutes=1.0,
        prediction_fieldnames=set(prediction),
        context_rows=[context],
    )

    assert report["safety_invariant_counters"] == "NOT_EVALUABLE"


def test_safety_context_uses_identity_to_detect_passenger_misattribution() -> None:
    prediction = _prediction(occupant_role="driver")
    context = _context_row(occupant_role="front_passenger")

    report = evaluate(
        truth_rows=_event_truth(),
        prediction_rows=[prediction],
        video_minutes=1.0,
        prediction_fieldnames=set(prediction),
        context_rows=[context],
    )

    counters = report["safety_invariant_counters"]
    assert counters != "NOT_EVALUABLE"
    assert counters["passenger_phone_violation_count"] == 1


def test_safety_context_fails_closed_for_unknown_behavior_state() -> None:
    prediction = _prediction()
    context = _context_row(phone_state="UNKNOWN")

    report = evaluate(
        truth_rows=_event_truth(),
        prediction_rows=[prediction],
        video_minutes=1.0,
        prediction_fieldnames=set(prediction),
        context_rows=[context],
    )

    assert report["safety_invariant_counters"] == "NOT_EVALUABLE"
