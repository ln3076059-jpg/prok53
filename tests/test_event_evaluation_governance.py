import csv
import json
from pathlib import Path

import pytest

from training.evaluate_events import evaluate_frozen, verify_evaluation_integrity
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
        "vehicle_id": "vehicle-1",
        "cabin_id": "cabin-1",
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
                "start_seconds": "1.2",
                "end_seconds": "2.1",
            }
        ],
        {"video_id", "event_type", "start_seconds", "end_seconds"},
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

    external_lock_original = external_lock_path.read_text(encoding="utf-8")
    external_lock_path.write_text(external_lock_original + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="external-test artifact SHA256 mismatch"):
        evaluate_frozen(
            truth_path,
            predictions_path,
            truth_lock_path,
            model_lock_path,
            output_path,
            video_minutes=1.0,
        )
    external_lock_path.write_text(external_lock_original, encoding="utf-8")

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
    report = evaluate_frozen(
        truth_path,
        predictions_path,
        truth_lock_path,
        model_lock_path,
        output_path,
        video_minutes=1.0,
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

    predictions_original = predictions_path.read_text(encoding="utf-8")
    predictions_path.write_text(predictions_original + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="predictions SHA256 changed"):
        verify_evaluation_integrity(output_path)
    predictions_path.write_text(predictions_original, encoding="utf-8")

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
    truth_path.write_text(truth_path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    predictions_path = tmp_path / "predictions.csv"
    _write_csv(
        predictions_path,
        [],
        {"video_id", "event_type", "start_seconds", "end_seconds"},
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
