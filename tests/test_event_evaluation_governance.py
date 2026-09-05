import csv
import json
from pathlib import Path

import pytest

from training.evaluate_events import evaluate_frozen, verify_evaluation_integrity
from training.freeze_context_ground_truth import (
    REQUIRED_COLUMNS as CONTEXT_REQUIRED_COLUMNS,
)
from training.freeze_context_ground_truth import freeze_context_ground_truth
from training.freeze_event_ground_truth import REQUIRED_COLUMNS, freeze_event_ground_truth


def _write_csv(path: Path, rows: list[dict[str, str]], fields: set[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=sorted(fields))
        writer.writeheader()
        writer.writerows(rows)


def _truth_row() -> dict[str, str]:
    return {
        "video_id": "external-video",
        "event_id": "event-1",
        "event_type": "PHONE",
        "start_seconds": "1.0",
        "end_seconds": "2.0",
        "occupant_id": "occupant-1",
        "vehicle_id": "vehicle-1",
        "cabin_id": "cabin-1",
        "inside_vehicle": "true",
        "outside_vehicle_person": "false",
        "motorcycle_flag": "false",
        "label": "PHONE_USE",
        "occupant_role": "driver",
        "visibility": "visible",
        "conditions": "daylight",
        "human_review_status": "APPROVED",
        "reviewer_id": "human-reviewer-1",
        "reviewer_type": "HUMAN",
        "reviewed_at": "2026-09-02T00:00:00Z",
        "adjudication_status": "FINAL",
        "notes": "independently reviewed",
    }


def _context_row(**overrides: str) -> dict[str, str]:
    row = {
        "video_id": "external-video",
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
        "phone_state": "PHONE_USE",
        "seatbelt_state": "FASTENED",
        "visibility": "visible",
        "conditions": "daylight",
        "human_review_status": "APPROVED",
        "reviewer_id": "human-reviewer-1",
        "reviewer_type": "HUMAN",
        "reviewed_at": "2026-09-02T00:00:00Z",
        "adjudication_status": "FINAL",
        "notes": "independently reviewed",
    }
    row.update(overrides)
    return row


def _freeze_context(tmp_path: Path, external_lock_path: Path) -> tuple[Path, Path]:
    context_path = tmp_path / "context.csv"
    context_lock_path = tmp_path / "context-lock.json"
    _write_csv(context_path, [_context_row()], CONTEXT_REQUIRED_COLUMNS)
    freeze_context_ground_truth(context_path, external_lock_path, context_lock_path)
    return context_path, context_lock_path


def _as_safety_context(row: dict[str, str]) -> dict[str, str]:
    return {
        "video_id": row["video_id"],
        "occupant_id": row["occupant_id"],
        "occupant_role": row.get("occupant_role", "driver"),
        "vehicle_id": row["vehicle_id"],
        "cabin_id": row["cabin_id"],
        "start_seconds": row["start_seconds"],
        "end_seconds": row["end_seconds"],
        "inside_vehicle": row.get("inside_vehicle", "true"),
        "outside_vehicle_person": row.get("outside_vehicle_person", "false"),
        "motorcycle_flag": row.get("motorcycle_flag", "false"),
        "phone_state": row.get("label", "NO_PHONE"),
        "seatbelt_state": row.get("label", "FASTENED"),
    }


def test_event_truth_freeze_requires_review_and_is_immutable(tmp_path):
    external_lock_path = tmp_path / "external-lock.json"
    external_lock_path.write_text(
        json.dumps(
            {
                "status": "FROZEN_EXTERNAL_TEST",
                "human_review_status": "ALL_APPROVED",
                "video_ids": ["external-video"],
            }
        ),
        encoding="utf-8",
    )
    truth_path = tmp_path / "truth.csv"
    output_path = tmp_path / "truth-lock.json"
    row = _truth_row()
    _write_csv(truth_path, [row], REQUIRED_COLUMNS)

    frozen = freeze_event_ground_truth(truth_path, external_lock_path, output_path)

    assert frozen["status"] == "FROZEN_EVENT_GROUND_TRUTH"
    assert frozen["event_type_counts"]["PHONE"] == 1
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        freeze_event_ground_truth(truth_path, external_lock_path, output_path)

    pending_path = tmp_path / "pending.csv"
    row["human_review_status"] = "PENDING"
    _write_csv(pending_path, [row], REQUIRED_COLUMNS)
    with pytest.raises(ValueError, match="human_review_status must be APPROVED"):
        freeze_event_ground_truth(pending_path, external_lock_path, tmp_path / "pending-lock.json")


def test_frozen_event_evaluation_requires_active_model_and_preserves_result(tmp_path):
    external_lock_path = tmp_path / "external-lock.json"
    external_lock_path.write_text(
        json.dumps(
            {
                "status": "FROZEN_EXTERNAL_TEST",
                "human_review_status": "ALL_APPROVED",
                "video_ids": ["external-video"],
            }
        ),
        encoding="utf-8",
    )
    truth_path = tmp_path / "truth.csv"
    _write_csv(truth_path, [_truth_row()], REQUIRED_COLUMNS)
    truth_lock_path = tmp_path / "truth-lock.json"
    freeze_event_ground_truth(truth_path, external_lock_path, truth_lock_path)

    predictions_path = tmp_path / "predictions.csv"
    _write_csv(
        predictions_path,
        [
            {
                "video_id": "external-video",
                "event_type": "PHONE",
                "occupant_id": "occupant-1",
                "occupant_role": "driver",
                "vehicle_id": "vehicle-1",
                "cabin_id": "cabin-1",
                "start_seconds": "1.2",
                "end_seconds": "2.1",
                "label": "PHONE_USE",
                "visibility": "clear",
                "outside_vehicle_person": "false",
                "motorcycle_flag": "false",
                "observation_count": "5"
            }
        ],
        {"video_id", "event_type", "occupant_id", "occupant_role", "vehicle_id", "cabin_id", "start_seconds", "end_seconds", "label", "visibility", "outside_vehicle_person", "motorcycle_flag", "observation_count"},
    )
    model_lock_path = tmp_path / "model-lock.json"
    model_lock = {
        "record_schema": "ROADWATCH_MODEL_VERSION_V2",
        "activation_state": "INACTIVE",
        "experiment_id": "V2_TEST",
        "locked_at": "2026-09-02T00:00:00+00:00",
        "weights_sha256": "a" * 64,
        "config_sha256": "b" * 64,
        "training_data_manifest_sha256": "c" * 64,
        "validation_metric_artifact": {"sha256": "d" * 64},
        "threshold_calibration_artifact": {"sha256": "e" * 64},
        "human_review_readiness_artifact": {"governed_training_ready": True},
        "code_commit": "abc123",
    }
    model_lock_path.write_text(json.dumps(model_lock), encoding="utf-8")
    output_path = tmp_path / "event-evaluation.json"

    external_lock_original = external_lock_path.read_bytes()
    external_lock_path.write_bytes(external_lock_original + b"mutation")
    with pytest.raises(ValueError, match="external-test artifact SHA256 mismatch"):
        evaluate_frozen(
            truth_path,
            predictions_path,
            truth_lock_path,
            model_lock_path,
            output_path,
            video_minutes=1.0,
        )
    external_lock_path.write_bytes(external_lock_original)

    with pytest.raises(ValueError, match="ACTIVE model lock"):
        evaluate_frozen(
            truth_path,
            predictions_path,
            truth_lock_path,
            model_lock_path,
            output_path,
            video_minutes=1.0,
        )

    model_lock["activation_state"] = "ACTIVE"
    model_lock_path.write_text(json.dumps(model_lock), encoding="utf-8")
    with pytest.raises(ValueError, match="separately frozen full-timeline context truth"):
        evaluate_frozen(
            truth_path,
            predictions_path,
            truth_lock_path,
            model_lock_path,
            output_path,
            video_minutes=1.0,
        )
    context_path, context_lock_path = _freeze_context(tmp_path, external_lock_path)
    report = evaluate_frozen(
        truth_path,
        predictions_path,
        truth_lock_path,
        model_lock_path,
        output_path,
        video_minutes=1.0,
        context_truth_path=context_path,
        context_truth_lock_path=context_lock_path,
    )

    assert report["status"] == "MEASURED_FROZEN_EXTERNAL_TEST"
    assert report["event_types"]["PHONE"]["false_events_per_hour"] == 0.0
    assert report["model_lock"]["experiment_id"] == "V2_TEST"
    assert (
        verify_evaluation_integrity(output_path)["status"]
        == "FROZEN_EVENT_EVALUATION_INTEGRITY_VERIFIED"
    )
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        evaluate_frozen(
            truth_path,
            predictions_path,
            truth_lock_path,
            model_lock_path,
            output_path,
            video_minutes=1.0,
        )

    predictions_original = predictions_path.read_bytes()
    predictions_path.write_bytes(predictions_original + b"mutation")
    with pytest.raises(ValueError, match="predictions SHA256 changed"):
        verify_evaluation_integrity(output_path)
    predictions_path.write_bytes(predictions_original)

    model_lock_path.write_text(
        json.dumps({**model_lock, "code_commit": "changed"}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="model lock SHA256 changed"):
        verify_evaluation_integrity(output_path)


def test_frozen_event_evaluation_rejects_truth_changed_after_freeze(tmp_path):
    external_lock_path = tmp_path / "external-lock.json"
    external_lock_path.write_text(
        json.dumps(
            {
                "status": "FROZEN_EXTERNAL_TEST",
                "human_review_status": "ALL_APPROVED",
                "video_ids": ["external-video"],
            }
        ),
        encoding="utf-8",
    )
    truth_path = tmp_path / "truth.csv"
    _write_csv(truth_path, [_truth_row()], REQUIRED_COLUMNS)
    truth_lock_path = tmp_path / "truth-lock.json"
    freeze_event_ground_truth(truth_path, external_lock_path, truth_lock_path)
    context_path, context_lock_path = _freeze_context(tmp_path, external_lock_path)
    truth_path.write_text(truth_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    predictions_path = tmp_path / "predictions.csv"
    _write_csv(
        predictions_path,
        [],
        {"video_id", "event_type", "occupant_role", "vehicle_id", "cabin_id", "start_seconds", "end_seconds"},
    )
    model_lock_path = tmp_path / "model-lock.json"
    model_lock_path.write_text(
        json.dumps(
            {
                "record_schema": "ROADWATCH_MODEL_VERSION_V2",
                "activation_state": "ACTIVE",
                "experiment_id": "V2_TEST",
                "locked_at": "2026-09-02T00:00:00+00:00",
                "weights_sha256": "a" * 64,
                "config_sha256": "b" * 64,
                "training_data_manifest_sha256": "c" * 64,
                "validation_metric_artifact": {"sha256": "d" * 64},
                "threshold_calibration_artifact": {"sha256": "e" * 64},
                "human_review_readiness_artifact": {"governed_training_ready": True},
                "code_commit": "abc123",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="truth SHA256"):
        evaluate_frozen(
            truth_path,
            predictions_path,
            truth_lock_path,
            model_lock_path,
            tmp_path / "evaluation.json",
            video_minutes=1.0,
        )
        
def test_evaluator_matching_requires_identity():
    from training.evaluate_events import evaluate

    truth_rows = [
        {
            "video_id": "vid1",
            "event_type": "NO_SEATBELT",
            "occupant_role": "rear_left",
            "vehicle_id": "veh1",
            "start_seconds": "10.0",
            "end_seconds": "15.0"
        }
    ]

    # 1. Matching role -> TP
    pred_match = [
        {
            "video_id": "vid1",
            "event_type": "NO_SEATBELT",
            "occupant_role": "rear_left",
            "vehicle_id": "veh1",
            "start_seconds": "11.0",
            "end_seconds": "14.0"
        }
    ]
    report = evaluate(truth_rows, pred_match, video_minutes=1.0)
    assert report["event_types"]["NO_SEATBELT"]["true_positives"] == 1
    assert report["event_types"]["NO_SEATBELT"]["false_positives"] == 0

    # 2. Mismatched role -> FP + FN
    pred_mismatch_role = [
        {
            "video_id": "vid1",
            "event_type": "NO_SEATBELT",
            "occupant_role": "rear_right",
            "vehicle_id": "veh1",
            "start_seconds": "11.0",
            "end_seconds": "14.0"
        }
    ]
    report2 = evaluate(truth_rows, pred_mismatch_role, video_minutes=1.0)
    assert report2["event_types"]["NO_SEATBELT"]["true_positives"] == 0
    assert report2["event_types"]["NO_SEATBELT"]["false_positives"] == 1
    assert report2["event_types"]["NO_SEATBELT"]["missed_events"] == 1

    # 3. Mismatched vehicle -> FP + FN
    pred_mismatch_veh = [
        {
            "video_id": "vid1",
            "event_type": "NO_SEATBELT",
            "occupant_role": "rear_left",
            "vehicle_id": "veh2",
            "start_seconds": "11.0",
            "end_seconds": "14.0"
        }
    ]
    report3 = evaluate(truth_rows, pred_mismatch_veh, video_minutes=1.0)
    assert report3["event_types"]["NO_SEATBELT"]["true_positives"] == 0
    assert report3["event_types"]["NO_SEATBELT"]["false_positives"] == 1
    assert report3["event_types"]["NO_SEATBELT"]["missed_events"] == 1

def test_evaluate_empty_predictions(tmp_path):
    from training.evaluate_events import evaluate_frozen, REQUIRED
    truth_path = tmp_path / "truth.csv"
    _write_csv(truth_path, [_truth_row()], REQUIRED_COLUMNS)
    external_lock_path = tmp_path / "external-lock.json"
    external_lock_path.write_text(json.dumps({"status": "FROZEN_EXTERNAL_TEST", "human_review_status": "ALL_APPROVED", "video_ids": ["external-video"]}), encoding="utf-8")
    truth_lock_path = tmp_path / "truth-lock.json"
    freeze_event_ground_truth(truth_path, external_lock_path, truth_lock_path)
    context_path, context_lock_path = _freeze_context(tmp_path, external_lock_path)
    
    predictions_path = tmp_path / "predictions.csv"
    _write_csv(predictions_path, [], REQUIRED | {"label", "visibility", "outside_vehicle_person", "motorcycle_flag", "observation_count"})
    
    model_lock_path = tmp_path / "model-lock.json"
    model_lock_path.write_text(
        json.dumps({
            "record_schema": "ROADWATCH_MODEL_VERSION_V2",
            "activation_state": "ACTIVE",
            "experiment_id": "V2_TEST",
            "locked_at": "2026-09-02T00:00:00+00:00",
            "weights_sha256": "a" * 64,
            "config_sha256": "b" * 64,
            "training_data_manifest_sha256": "c" * 64,
            "validation_metric_artifact": {"sha256": "d" * 64},
            "threshold_calibration_artifact": {"sha256": "e" * 64},
            "human_review_readiness_artifact": {"governed_training_ready": True},
            "code_commit": "abc123"
        }),
        encoding="utf-8"
    )
    
    output_path = tmp_path / "evaluation.json"
    report = evaluate_frozen(
        truth_path,
        predictions_path,
        truth_lock_path,
        model_lock_path,
        output_path,
        video_minutes=1.0,
        context_truth_path=context_path,
        context_truth_lock_path=context_lock_path,
    )
    assert report["event_types"]["PHONE"]["true_positives"] == 0
    assert report["event_types"]["PHONE"]["false_positives"] == 0
    assert report["event_types"]["PHONE"]["missed_events"] == 1
    assert report["event_types"]["PHONE"]["recall"] == 0.0

def test_evaluator_order_independent_matching():
    from training.evaluate_events import evaluate
    
    truth_rows = [
        {
            "video_id": "vid1",
            "event_type": "PHONE",
            "occupant_role": "driver",
            "start_seconds": "0.0",
            "end_seconds": "10.0"
        },
        {
            "video_id": "vid1",
            "event_type": "PHONE",
            "occupant_role": "driver",
            "start_seconds": "8.0",
            "end_seconds": "18.0"
        }
    ]
    
    pred_p1 = {
        "video_id": "vid1",
        "event_type": "PHONE",
        "occupant_role": "driver",
        "start_seconds": "1.0",
        "end_seconds": "9.0"
    } # Overlaps both truth A and truth B
    
    pred_p2 = {
        "video_id": "vid1",
        "event_type": "PHONE",
        "occupant_role": "driver",
        "start_seconds": "0.0",
        "end_seconds": "1.0"
    } # Overlaps only truth A
    
    # 1. Order P1, P2
    report1 = evaluate(truth_rows, [pred_p1, pred_p2], video_minutes=1.0, tolerance_seconds=0.5)
    
    # 2. Order P2, P1
    report2 = evaluate(truth_rows, [pred_p2, pred_p1], video_minutes=1.0, tolerance_seconds=0.5)
    
    # Both should optimally assign P2->A and P1->B resulting in 2 TP, 0 FP, 0 FN
    assert report1["event_types"]["PHONE"]["true_positives"] == 2
    assert report1["event_types"]["PHONE"]["false_positives"] == 0
    assert report1["event_types"]["PHONE"]["missed_events"] == 0
    
    assert report2["event_types"]["PHONE"]["true_positives"] == 2
    assert report2["event_types"]["PHONE"]["false_positives"] == 0
    assert report2["event_types"]["PHONE"]["missed_events"] == 0

def test_evaluator_safety_counters():
    from training.evaluate_events import evaluate
    
    # 1. Missing metadata test
    report_missing = evaluate(
        truth_rows=[],
        prediction_rows=[{"event_type": "PHONE"}],
        video_minutes=1.0
    )
    assert report_missing["safety_invariant_counters"] == "NOT_EVALUABLE"
    
    # 2. Fully populated test
    truth_rows = [
        {"video_id": "v1", "event_type": "PHONE", "start_seconds": "0", "end_seconds": "1", "occupant_role": "front_passenger", "vehicle_id": "v1", "cabin_id": "c1"},
        {"video_id": "v1", "event_type": "PHONE", "start_seconds": "2", "end_seconds": "5", "label": "MOUNTED_OR_STATIC_PHONE", "vehicle_id": "v1", "cabin_id": "c1"},
        {"video_id": "v1", "event_type": "NO_SEATBELT", "start_seconds": "6", "end_seconds": "7", "outside_vehicle_person": "true", "vehicle_id": "v1", "cabin_id": "c1"},
        {"video_id": "v1", "event_type": "PHONE", "start_seconds": "8", "end_seconds": "9", "vehicle_id": "v1", "cabin_id": "c1"},
        {"video_id": "v1", "event_type": "NO_SEATBELT", "start_seconds": "10", "end_seconds": "11", "motorcycle_flag": "true", "vehicle_id": "v1", "cabin_id": "c1"},
        {"video_id": "v1", "event_type": "NO_SEATBELT", "start_seconds": "12", "end_seconds": "13", "vehicle_id": "v1", "cabin_id": "c1"},
        {"video_id": "v1", "event_type": "PHONE", "start_seconds": "15", "end_seconds": "16", "vehicle_id": "v1", "cabin_id": "c1"}
    ]
    
    prediction_rows = [
        # passenger_phone_violation
        {"video_id": "v1", "event_type": "PHONE", "occupant_role": "front_passenger", "start_seconds": "0", "end_seconds": "1", "label": "PHONE_USE", "visibility": "clear", "outside_vehicle_person": "false", "motorcycle_flag": "false", "vehicle_id": "v1", "cabin_id": "c1", "start_frame": "0", "end_frame": "30"},
        # mounted_phone_violation (by label)
        {"video_id": "v1", "event_type": "PHONE", "occupant_role": "driver", "label": "MOUNTED_OR_STATIC_PHONE", "start_seconds": "2", "end_seconds": "3", "visibility": "clear", "outside_vehicle_person": "false", "motorcycle_flag": "false", "vehicle_id": "v1", "cabin_id": "c1", "start_frame": "60", "end_frame": "90"},
        # mounted_phone_violation (by visibility)
        {"video_id": "v1", "event_type": "PHONE", "occupant_role": "driver", "visibility": "mounted", "start_seconds": "4", "end_seconds": "5", "label": "PHONE_USE", "outside_vehicle_person": "false", "motorcycle_flag": "false", "vehicle_id": "v1", "cabin_id": "c1", "start_frame": "120", "end_frame": "150"},
        # outside_person_violation
        {"video_id": "v1", "event_type": "NO_SEATBELT", "occupant_role": "driver", "outside_vehicle_person": "True", "start_seconds": "6", "end_seconds": "7", "label": "UNFASTENED", "visibility": "clear", "motorcycle_flag": "false", "vehicle_id": "v1", "cabin_id": "c1", "start_frame": "180", "end_frame": "210"},
        # unknown_role_phone_violation
        {"video_id": "v1", "event_type": "PHONE", "occupant_role": "unknown", "start_seconds": "8", "end_seconds": "9", "label": "PHONE_USE", "visibility": "clear", "outside_vehicle_person": "false", "motorcycle_flag": "false", "vehicle_id": "v1", "cabin_id": "c1", "start_frame": "240", "end_frame": "270"},
        # motorcycle_seatbelt_violation
        {"video_id": "v1", "event_type": "NO_SEATBELT", "occupant_role": "driver", "motorcycle_flag": "True", "start_seconds": "10", "end_seconds": "11", "label": "UNFASTENED", "visibility": "clear", "outside_vehicle_person": "false", "vehicle_id": "v1", "cabin_id": "c1", "start_frame": "300", "end_frame": "330"},
        # unknown_belt_violation
        {"video_id": "v1", "event_type": "NO_SEATBELT", "occupant_role": "unknown", "start_seconds": "12", "end_seconds": "13", "label": "UNFASTENED", "visibility": "clear", "outside_vehicle_person": "false", "motorcycle_flag": "false", "vehicle_id": "v1", "cabin_id": "c1", "start_frame": "360", "end_frame": "390"},
        # single_frame_violation
        {"video_id": "v1", "event_type": "PHONE", "occupant_role": "driver", "start_seconds": "15", "end_seconds": "15.033", "label": "PHONE_USE", "visibility": "clear", "outside_vehicle_person": "false", "motorcycle_flag": "false", "vehicle_id": "v1", "cabin_id": "c1", "start_frame": "450", "end_frame": "450"},
    ]

    context_roles = [
        "front_passenger", "driver", "driver", "driver",
        "driver", "driver", "driver", "driver",
    ]
    context_rows = []
    for index, (prediction, context_role) in enumerate(
        zip(prediction_rows, context_roles, strict=True)
    ):
        occupant_id = f"occupant-{index}"
        prediction["occupant_id"] = occupant_id
        context_rows.append(
            {
                "video_id": "v1",
                "occupant_id": occupant_id,
                "occupant_role": context_role,
                "vehicle_id": "v1",
                "cabin_id": "c1",
                "start_seconds": prediction["start_seconds"],
                "end_seconds": prediction["end_seconds"],
                "inside_vehicle": "true",
                "outside_vehicle_person": "true" if index == 3 else "false",
                "motorcycle_flag": "true" if index == 5 else "false",
                "phone_state": (
                    "MOUNTED_OR_STATIC_PHONE" if index in {1, 2} else "NO_PHONE"
                ),
                "seatbelt_state": "FASTENED",
            }
        )

    report = evaluate(
        truth_rows=truth_rows,
        prediction_rows=prediction_rows,
        video_minutes=1.0,
        context_rows=context_rows,
    )
    counters = report["safety_invariant_counters"]
    assert counters != "NOT_EVALUABLE"
    assert counters["passenger_phone_violation_count"] == 1
    assert counters["mounted_phone_violation_count"] == 2
    assert counters["outside_person_violation_count"] == 1
    assert counters["unknown_role_phone_violation_count"] == 1
    assert counters["motorcycle_seatbelt_violation_count"] == 1
    assert counters["unknown_belt_violation_count"] == 1
    assert counters["single_frame_violation_count"] == 1

def test_evaluator_ambiguous_overlap():
    from training.evaluate_events import evaluate
    
    # Truth has two overlapping events but different labels
    truth_rows = [
        {"video_id": "v1", "event_type": "PHONE", "occupant_id": "occupant-1", "start_seconds": "10", "end_seconds": "15", "occupant_role": "driver", "label": "PHONE_USE", "vehicle_id": "veh1", "cabin_id": "cab1", "inside_vehicle": "true", "outside_vehicle_person": "false", "motorcycle_flag": "false"},
        {"video_id": "v1", "event_type": "PHONE", "occupant_id": "occupant-1", "start_seconds": "12", "end_seconds": "14", "occupant_role": "driver", "label": "MOUNTED_OR_STATIC_PHONE", "vehicle_id": "veh1", "cabin_id": "cab1", "inside_vehicle": "true", "outside_vehicle_person": "false", "motorcycle_flag": "false"}
    ]
    
    # Prediction overlaps both
    prediction_rows = [
        {"video_id": "v1", "event_type": "PHONE", "occupant_id": "occupant-1", "occupant_role": "driver", "start_seconds": "11", "end_seconds": "14", "label": "PHONE_USE", "visibility": "clear", "outside_vehicle_person": "false", "motorcycle_flag": "false", "vehicle_id": "veh1", "cabin_id": "cab1", "start_frame": "330", "end_frame": "420"}
    ]
    
    report = evaluate(
        truth_rows=truth_rows,
        prediction_rows=prediction_rows,
        video_minutes=1.0,
        context_rows=[_as_safety_context(row) for row in truth_rows],
    )
    assert report["safety_invariant_counters"] == "NOT_EVALUABLE"

def test_evaluator_safety_counters_fail_close_missing_context():
    from training.evaluate_events import evaluate
    
    # Truth has a single event
    truth_rows = [
        {"video_id": "v1", "event_type": "PHONE", "occupant_id": "occupant-1", "start_seconds": "10", "end_seconds": "15", "occupant_role": "driver", "label": "PHONE_USE", "vehicle_id": "veh_A", "cabin_id": "cab1", "inside_vehicle": "true", "outside_vehicle_person": "false", "motorcycle_flag": "false"}
    ]
    
    # Prediction 1: overlaps but has wrong vehicle_id (veh_B vs veh_A) -> incompatible context -> NOT_EVALUABLE
    pred_incompatible = [
        {"video_id": "v1", "event_type": "PHONE", "occupant_id": "occupant-1", "occupant_role": "driver", "start_seconds": "11", "end_seconds": "14", "label": "PHONE_USE", "visibility": "clear", "outside_vehicle_person": "false", "motorcycle_flag": "false", "vehicle_id": "veh_B", "cabin_id": "cab1", "start_frame": "330", "end_frame": "420"}
    ]
    
    context_rows = [_as_safety_context(row) for row in truth_rows]
    report1 = evaluate(
        truth_rows=truth_rows,
        prediction_rows=pred_incompatible,
        video_minutes=1.0,
        context_rows=context_rows,
    )
    assert report1["safety_invariant_counters"] == "NOT_EVALUABLE"
    
    # Prediction 2: compatible vehicle, but completely misses the temporal window -> no context -> NOT_EVALUABLE
    pred_no_overlap = [
        {"video_id": "v1", "event_type": "PHONE", "occupant_id": "occupant-1", "occupant_role": "driver", "start_seconds": "20", "end_seconds": "25", "label": "PHONE_USE", "visibility": "clear", "outside_vehicle_person": "false", "motorcycle_flag": "false", "vehicle_id": "veh_A", "cabin_id": "cab1", "start_frame": "600", "end_frame": "750"}
    ]
    
    report2 = evaluate(
        truth_rows=truth_rows,
        prediction_rows=pred_no_overlap,
        video_minutes=1.0,
        context_rows=context_rows,
    )
    assert report2["safety_invariant_counters"] == "NOT_EVALUABLE"
