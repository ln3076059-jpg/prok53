from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha256_file

REQUIRED = {"video_id", "event_type", "occupant_role", "vehicle_id", "cabin_id", "start_seconds", "end_seconds"}


def _read(path: Path) -> tuple[list[dict], set[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    return rows, set(reader.fieldnames or [])


def _overlap(a: dict, b: dict, tolerance: float) -> bool:
    if a["video_id"] != b["video_id"] or a["event_type"] != b["event_type"]:
        return False
    if a.get("occupant_role") != b.get("occupant_role"):
        return False
    for field in ("vehicle_id", "cabin_id"):
        val_a = a.get(field)
        val_b = b.get(field)
        if val_a and val_b and val_a != "UNKNOWN" and val_b != "UNKNOWN" and val_a != val_b:
            return False
    a_start, a_end = float(a["start_seconds"]), float(a["end_seconds"])
    b_start, b_end = float(b["start_seconds"]), float(b["end_seconds"])
    return b_start <= a_end + tolerance and a_start <= b_end + tolerance


def evaluate(
    truth_rows: list[dict],
    prediction_rows: list[dict],
    video_minutes: float,
    tolerance_seconds: float = 0.5,
    prediction_fieldnames: set[str] | None = None,
) -> dict:
    from scipy.optimize import linear_sum_assignment
    import numpy as np

    matches: list[tuple[int, int]] = []
    if truth_rows and prediction_rows:
        cost_matrix = np.full((len(truth_rows), len(prediction_rows)), 1e9)
        for t_idx, truth in enumerate(truth_rows):
            for p_idx, pred in enumerate(prediction_rows):
                if _overlap(truth, pred, tolerance_seconds):
                    cost_matrix[t_idx, p_idx] = abs(
                        float(truth["start_seconds"]) - float(pred["start_seconds"])
                    )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind):
            if cost_matrix[r, c] < 1e8:
                matches.append((int(r), int(c)))

    matched_predictions = {p for t, p in matches}
    
    duplicate_predictions: set[int] = set()
    for p_idx, pred in enumerate(prediction_rows):
        if p_idx not in matched_predictions:
            for t_idx, truth in enumerate(truth_rows):
                if _overlap(truth, pred, tolerance_seconds):
                    duplicate_predictions.add(p_idx)
                    break

    report = {"status": "MEASURED", "event_types": {}}
    event_types = sorted({row["event_type"] for row in truth_rows + prediction_rows})
    for event_type in event_types:
        truth_indices = {i for i, row in enumerate(truth_rows) if row["event_type"] == event_type}
        prediction_indices = {
            i for i, row in enumerate(prediction_rows) if row["event_type"] == event_type
        }
        type_matches = [
            (t, p) for t, p in matches if t in truth_indices and p in prediction_indices
        ]
        tp = len(type_matches)
        fp = len(prediction_indices) - tp
        fn = len(truth_indices) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        delays = [
            float(prediction_rows[p]["start_seconds"]) - float(truth_rows[t]["start_seconds"])
            for t, p in type_matches
        ]
        type_duplicates = [p for p in duplicate_predictions if p in prediction_indices]
        
        report["event_types"][event_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "missed_events": fn,
            "false_alarms_per_minute": fp / video_minutes if video_minutes > 0 else None,
            "missed_events_per_minute": fn / video_minutes if video_minutes > 0 else None,
            "mean_time_to_detection_seconds": sum(delays) / len(delays) if delays else None,
            "duplicate_alerts_per_event": len(type_duplicates) / tp if tp > 0 else None,
            # Backwards compatibility
            "false_events_per_minute": fp / video_minutes if video_minutes > 0 else None,
            "false_events_per_hour": fp / video_minutes * 60 if video_minutes > 0 else None,
            "event_fragmentation_count": len(type_duplicates),
        }

    if prediction_fieldnames is None:
        prediction_fieldnames = set()
        for p in prediction_rows:
            prediction_fieldnames.update(p.keys())

    has_frame_info = "start_frame" in prediction_fieldnames and "end_frame" in prediction_fieldnames
    has_obs_info = "observation_count" in prediction_fieldnames
    safety_requirements = {"label", "visibility", "outside_vehicle_person", "motorcycle_flag"}
    
    def _is_valid_row(p: dict) -> bool:
        role = p.get("occupant_role", "")
        if role not in {"driver", "front_passenger", "rear_left", "rear_center", "rear_right", "unknown"}:
            return False
        evt = p.get("event_type", "")
        if evt not in {"PHONE", "NO_SEATBELT"}:
            return False
        if p.get("outside_vehicle_person", "").lower() not in {"true", "false"}:
            return False
        if p.get("motorcycle_flag", "").lower() not in {"true", "false"}:
            return False
        if p.get("vehicle_id", "") == "" or p.get("cabin_id", "") == "":
            return False
        try:
            start = float(p.get("start_seconds", -1))
            end = float(p.get("end_seconds", -1))
            if not (0 <= start <= end) or start == float('inf') or end == float('inf'):
                return False
        except ValueError:
            return False
        label = p.get("label", "")
        if evt == "PHONE" and label not in {"PHONE_USE", "PHONE_PRESENT_NOT_USED", "MOUNTED_OR_STATIC_PHONE", "NO_PHONE", "UNKNOWN"}:
            return False
        if evt == "NO_SEATBELT" and label not in {"FASTENED", "UNFASTENED", "UNCERTAIN_OR_OCCLUDED", "NOT_APPLICABLE"}:
            return False
            
        if has_frame_info:
            try:
                sf = int(p["start_frame"])
                ef = int(p["end_frame"])
                if sf < 0 or ef < sf:
                    return False
            except ValueError:
                return False
        if has_obs_info:
            try:
                if int(p["observation_count"]) < 1:
                    return False
            except ValueError:
                return False
                
        return True

    if not safety_requirements.issubset(prediction_fieldnames) or not (has_frame_info or has_obs_info):
        counters = "NOT_EVALUABLE"
    elif not all(_is_valid_row(p) for p in prediction_rows):
        counters = "NOT_EVALUABLE"
    else:
        counters = {
            "passenger_phone_violation_count": 0,
            "mounted_phone_violation_count": 0,
            "outside_person_violation_count": 0,
            "unknown_role_phone_violation_count": 0,
            "motorcycle_seatbelt_violation_count": 0,
            "unknown_belt_violation_count": 0,
            "single_frame_violation_count": 0,
        }
        
        for p in prediction_rows:
            evt = p["event_type"]
            role = p.get("occupant_role", "")
            if evt == "PHONE" and role in ("front_passenger", "rear_left", "rear_center", "rear_right"):
                counters["passenger_phone_violation_count"] += 1
            if evt == "PHONE" and (p.get("visibility") == "mounted" or p.get("label") == "MOUNTED_OR_STATIC_PHONE"):
                counters["mounted_phone_violation_count"] += 1
            if str(p.get("outside_vehicle_person")).lower() == "true":
                counters["outside_person_violation_count"] += 1
            if evt == "PHONE" and role == "unknown":
                counters["unknown_role_phone_violation_count"] += 1
            if evt == "NO_SEATBELT" and str(p.get("motorcycle_flag")).lower() == "true":
                counters["motorcycle_seatbelt_violation_count"] += 1
            if evt == "NO_SEATBELT" and role == "unknown":
                counters["unknown_belt_violation_count"] += 1
                
            if has_frame_info:
                try:
                    if int(p["start_frame"]) == int(p["end_frame"]):
                        counters["single_frame_violation_count"] += 1
                except ValueError:
                    pass
            elif has_obs_info:
                try:
                    if int(p["observation_count"]) == 1:
                        counters["single_frame_violation_count"] += 1
                except ValueError:
                    pass

    report["safety_invariant_counters"] = counters
    report["duplicate_events"] = len(duplicate_predictions)
    report["event_fragmentation_count"] = len(duplicate_predictions)
    report["video_minutes"] = video_minutes
    report["matching_tolerance_seconds"] = tolerance_seconds
    return report


def evaluate_frozen(
    truth_path: Path,
    prediction_path: Path,
    ground_truth_lock_path: Path,
    model_lock_path: Path,
    output_path: Path,
    video_minutes: float,
    tolerance_seconds: float = 0.5,
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen event evaluation: {output_path}")
    if video_minutes <= 0:
        raise ValueError("frozen event evaluation requires positive --video-minutes")
    ground_truth_lock = json.loads(ground_truth_lock_path.read_text(encoding="utf-8"))
    if ground_truth_lock.get("status") != "FROZEN_EVENT_GROUND_TRUTH":
        raise ValueError("ground-truth artifact is not FROZEN_EVENT_GROUND_TRUTH")
    if ground_truth_lock.get("human_review_status") != "ALL_APPROVED":
        raise ValueError("ground-truth artifact is not fully human-approved")
    if ground_truth_lock.get("adjudication_status") != "ALL_FINAL":
        raise ValueError("ground-truth artifact is not fully adjudicated")
    if ground_truth_lock.get("truth_csv", {}).get("sha256") != sha256_file(truth_path):
        raise ValueError("event truth SHA256 does not match its frozen artifact")

    external_record = ground_truth_lock.get("external_test_lock", {})
    external_lock_path = Path(str(external_record.get("path", "")))
    if not external_lock_path.is_file():
        raise ValueError("frozen external-test artifact referenced by event truth is missing")
    if external_record.get("sha256") != sha256_file(external_lock_path):
        raise ValueError("frozen external-test artifact SHA256 mismatch")
    external_lock = json.loads(external_lock_path.read_text(encoding="utf-8"))
    if external_lock.get("status") != "FROZEN_EXTERNAL_TEST":
        raise ValueError("referenced external-test artifact is not frozen")

    model_lock = json.loads(model_lock_path.read_text(encoding="utf-8"))
    if model_lock.get("record_schema") != "ROADWATCH_MODEL_VERSION_V2":
        raise ValueError("model lock is not a V2 model-lock artifact")
    if model_lock.get("activation_state") != "ACTIVE":
        raise ValueError("frozen event evaluation requires an ACTIVE model lock")
    required_model_lock_fields = {
        "experiment_id",
        "locked_at",
        "weights_sha256",
        "config_sha256",
        "training_data_manifest_sha256",
        "validation_metric_artifact",
        "threshold_calibration_artifact",
        "human_review_readiness_artifact",
        "code_commit",
    }
    missing_model_lock_fields = required_model_lock_fields - set(model_lock)
    if missing_model_lock_fields:
        raise ValueError(
            f"model lock lacks governed provenance: {sorted(missing_model_lock_fields)}"
        )
    calibration_record = model_lock["threshold_calibration_artifact"]
    if not isinstance(calibration_record, dict) or not calibration_record.get("sha256"):
        raise ValueError("model lock has no threshold calibration artifact")
    readiness_record = model_lock["human_review_readiness_artifact"]
    if not isinstance(readiness_record, dict) or (
        readiness_record.get("governed_training_ready") is not True
    ):
        raise ValueError("model lock does not record governed human-review readiness")

    truth_rows, _ = _read(truth_path)
    prediction_rows, pred_fields = _read(prediction_path)
    report = evaluate(
        truth_rows,
        prediction_rows,
        video_minutes,
        tolerance_seconds,
        prediction_fieldnames=pred_fields,
    )
    if report.get("safety_invariant_counters") == "NOT_EVALUABLE":
        raise ValueError("frozen event evaluation requires evaluable safety metadata")
        
    report.update(
        {
            "status": "MEASURED_FROZEN_EXTERNAL_TEST",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "ground_truth": {
                "path": str(truth_path),
                "sha256": sha256_file(truth_path),
                "lock_path": str(ground_truth_lock_path),
                "lock_sha256": sha256_file(ground_truth_lock_path),
            },
            "predictions": {
                "path": str(prediction_path),
                "sha256": sha256_file(prediction_path),
            },
            "model_lock": {
                "path": str(model_lock_path),
                "sha256": sha256_file(model_lock_path),
                "experiment_id": model_lock.get("experiment_id"),
                "code_commit": model_lock.get("code_commit"),
            },
            "external_test_lock": external_record,
            "scientific_claim": "FROZEN_EVENT_METRICS_FOR_THIS_LOCKED_MODEL_ONLY",
        }
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return report


def verify_evaluation_integrity(report_path: Path) -> dict:
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if report.get("status") != "MEASURED_FROZEN_EXTERNAL_TEST":
        raise ValueError("event evaluation is not a frozen external-test result")
    artifacts = {
        "ground truth": report.get("ground_truth", {}),
        "ground-truth lock": {
            "path": report.get("ground_truth", {}).get("lock_path"),
            "sha256": report.get("ground_truth", {}).get("lock_sha256"),
        },
        "predictions": report.get("predictions", {}),
        "model lock": report.get("model_lock", {}),
        "external-test lock": report.get("external_test_lock", {}),
    }
    verified: dict[str, str] = {}
    for name, record in artifacts.items():
        path = Path(str(record.get("path", "")))
        expected_hash = record.get("sha256")
        if not path.is_file():
            raise ValueError(f"{name} referenced by event evaluation is missing")
        actual_hash = sha256_file(path)
        if not expected_hash or actual_hash != expected_hash:
            raise ValueError(f"{name} SHA256 changed after frozen evaluation")
        verified[name] = actual_hash
    return {
        "status": "FROZEN_EVENT_EVALUATION_INTEGRITY_VERIFIED",
        "report_path": str(report_path),
        "report_sha256": sha256_file(report_path),
        "verified_artifacts": verified,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate event-level predictions against frozen human ground truth"
    )
    parser.add_argument("truth", type=Path, nargs="?")
    parser.add_argument("predictions", type=Path, nargs="?")
    parser.add_argument("--video-minutes", type=float, default=0.0)
    parser.add_argument("--tolerance-seconds", type=float, default=0.5)
    parser.add_argument(
        "--verify-existing",
        type=Path,
        help="Verify that every artifact bound to an existing frozen result is unchanged",
    )
    parser.add_argument(
        "--ground-truth-lock",
        type=Path,
        default=Path("datasets/manifests/v2_event_truth_frozen.json"),
    )
    parser.add_argument(
        "--model-lock",
        type=Path,
        default=Path("reports/model_lock_v2.json"),
    )
    parser.add_argument("--output", type=Path, default=Path("reports/event_evaluation.json"))
    args = parser.parse_args()
    if args.verify_existing:
        print(json.dumps(verify_evaluation_integrity(args.verify_existing), indent=2))
        return
    if not args.truth or not args.predictions:
        report = {
            "status": "NOT_RUN",
            "reason": "human ground truth and predictions are required",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    report = evaluate_frozen(
        args.truth,
        args.predictions,
        args.ground_truth_lock,
        args.model_lock,
        args.output,
        args.video_minutes,
        args.tolerance_seconds,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
