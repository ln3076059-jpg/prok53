from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha256_file

REQUIRED = {
    "video_id",
    "event_type",
    "occupant_id",
    "occupant_role",
    "vehicle_id",
    "cabin_id",
    "start_seconds",
    "end_seconds",
}
CONTEXT_REQUIRED = {
    "video_id",
    "occupant_id",
    "occupant_role",
    "vehicle_id",
    "cabin_id",
    "start_seconds",
    "end_seconds",
    "inside_vehicle",
    "outside_vehicle_person",
    "motorcycle_flag",
    "phone_state",
    "seatbelt_state",
}


def _read(path: Path, required: set[str] | None = None) -> tuple[list[dict], set[str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = (required or REQUIRED) - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    return rows, set(reader.fieldnames or [])


def _overlap(a: dict, b: dict, tolerance: float) -> bool:
    if a["video_id"] != b["video_id"] or a["event_type"] != b["event_type"]:
        return False
    if a.get("occupant_id") != b.get("occupant_id"):
        return False
    if a.get("occupant_role") != b.get("occupant_role"):
        return False
    for field in ("vehicle_id", "cabin_id"):
        if a.get(field) != b.get(field):
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
    context_rows: list[dict] | None = None,
) -> dict:
    import numpy as np
    from scipy.optimize import linear_sum_assignment

    def _is_detectable(t: dict) -> bool:
        evt = t.get("event_type", "")
        label = t.get("label", "PHONE_USE" if evt == "PHONE" else "UNFASTENED")
        role = t.get("occupant_role", "driver")
        inside = str(t.get("inside_vehicle", "true")).lower() == "true"
        outside = str(t.get("outside_vehicle_person", "false")).lower() == "true"
        moto = str(t.get("motorcycle_flag", "false")).lower() == "true"

        if evt == "PHONE":
            return label == "PHONE_USE" and role == "driver" and inside and not outside
        elif evt == "NO_SEATBELT":
            return (
                label == "UNFASTENED"
                and not moto
                and inside
                and not outside
                and role in {"driver", "front_passenger", "rear_left", "rear_center", "rear_right"}
            )
        return False

    detectable_truth_rows = [t for t in truth_rows if _is_detectable(t)]

    matches: list[tuple[int, int]] = []
    if detectable_truth_rows and prediction_rows:
        cost_matrix = np.full((len(detectable_truth_rows), len(prediction_rows)), 1e9)
        for t_idx, truth in enumerate(detectable_truth_rows):
            for p_idx, pred in enumerate(prediction_rows):
                if _overlap(truth, pred, tolerance_seconds):
                    cost_matrix[t_idx, p_idx] = abs(
                        float(truth["start_seconds"]) - float(pred["start_seconds"])
                    )
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for r, c in zip(row_ind, col_ind, strict=False):
            if cost_matrix[r, c] < 1e8:
                matches.append((int(r), int(c)))

    matched_predictions = {p for t, p in matches}

    duplicate_predictions: set[int] = set()
    for p_idx, pred in enumerate(prediction_rows):
        if p_idx not in matched_predictions:
            for truth in detectable_truth_rows:
                if _overlap(truth, pred, tolerance_seconds):
                    duplicate_predictions.add(p_idx)
                    break

    report = {"status": "MEASURED", "event_types": {}}
    event_types = sorted({row["event_type"] for row in detectable_truth_rows + prediction_rows})
    for event_type in event_types:
        truth_indices = {
            i for i, row in enumerate(detectable_truth_rows) if row["event_type"] == event_type
        }
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
            float(prediction_rows[p]["start_seconds"])
            - float(detectable_truth_rows[t]["start_seconds"])
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
        if role not in {
            "driver",
            "front_passenger",
            "rear_left",
            "rear_center",
            "rear_right",
            "unknown",
        }:
            return False
        evt = p.get("event_type", "")
        if evt not in {"PHONE", "NO_SEATBELT"}:
            return False
        if p.get("outside_vehicle_person", "").lower() not in {"true", "false", ""}:
            return False
        if p.get("motorcycle_flag", "").lower() not in {"true", "false", ""}:
            return False
        for field in ("occupant_id", "vehicle_id", "cabin_id"):
            if not p.get(field) or p[field].upper() == "UNKNOWN":
                return False
        try:
            start = float(p.get("start_seconds", -1))
            end = float(p.get("end_seconds", -1))
            if not (0 <= start <= end) or start == float("inf") or end == float("inf"):
                return False
        except ValueError:
            return False
        label = p.get("label", "")
        if evt == "PHONE" and label not in {
            "PHONE_USE",
            "PHONE_PRESENT_NOT_USED",
            "MOUNTED_OR_STATIC_PHONE",
            "NO_PHONE",
            "UNKNOWN",
        }:
            return False
        if evt == "NO_SEATBELT" and label not in {
            "FASTENED",
            "UNFASTENED",
            "UNCERTAIN_OR_OCCLUDED",
            "NOT_APPLICABLE",
        }:
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

    if context_rows is None:
        counters = "NOT_EVALUABLE"
    elif not safety_requirements.issubset(prediction_fieldnames) or not (
        has_frame_info or has_obs_info
    ):
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

            p_start = float(p["start_seconds"])
            p_end = float(p["end_seconds"])
            matching_context = [
                context
                for context in context_rows
                if context["video_id"] == p["video_id"]
                and context.get("occupant_id") == p.get("occupant_id")
                and context.get("vehicle_id") == p.get("vehicle_id")
                and context.get("cabin_id") == p.get("cabin_id")
                and float(context["end_seconds"]) > p_start
                and float(context["start_seconds"]) < p_end
            ]
            if p_start == p_end:
                matching_context = [
                    context
                    for context in context_rows
                    if context["video_id"] == p["video_id"]
                    and context.get("occupant_id") == p.get("occupant_id")
                    and context.get("vehicle_id") == p.get("vehicle_id")
                    and context.get("cabin_id") == p.get("cabin_id")
                    and float(context["start_seconds"]) <= p_start
                    and p_start < float(context["end_seconds"])
                ]

            if not matching_context:
                counters = "NOT_EVALUABLE"
                break

            ordered_context = sorted(
                matching_context, key=lambda context: float(context["start_seconds"])
            )
            cursor = p_start
            for context in ordered_context:
                start = max(p_start, float(context["start_seconds"]))
                end = min(p_end, float(context["end_seconds"]))
                if start > cursor + 1e-6:
                    counters = "NOT_EVALUABLE"
                    break
                cursor = max(cursor, end)
            if counters == "NOT_EVALUABLE":
                break
            if cursor < p_end - 1e-6:
                counters = "NOT_EVALUABLE"
                break

            unique_contexts = {
                (
                    context.get("occupant_id"),
                    context.get("vehicle_id"),
                    context.get("cabin_id"),
                    context.get("occupant_role"),
                    str(context.get("inside_vehicle", "")).lower(),
                    str(context.get("outside_vehicle_person", "")).lower(),
                    str(context.get("motorcycle_flag", "")).lower(),
                    context.get("phone_state"),
                    context.get("seatbelt_state"),
                )
                for context in matching_context
            }
            if len(unique_contexts) != 1:
                counters = "NOT_EVALUABLE"
                break

            context = matching_context[0]
            if context.get("occupant_role") == "unknown":
                counters = "NOT_EVALUABLE"
                break
            if evt == "PHONE" and context.get("phone_state") == "UNKNOWN":
                counters = "NOT_EVALUABLE"
                break
            if evt == "NO_SEATBELT" and context.get("seatbelt_state") in {
                "UNCERTAIN_OR_OCCLUDED",
                "",
                None,
            }:
                counters = "NOT_EVALUABLE"
                break

            if evt == "PHONE" and context.get("occupant_role") in (
                "front_passenger",
                "rear_left",
                "rear_center",
                "rear_right",
            ):
                counters["passenger_phone_violation_count"] += 1

            if evt == "PHONE" and context.get("phone_state") == "MOUNTED_OR_STATIC_PHONE":
                counters["mounted_phone_violation_count"] += 1

            if str(context.get("outside_vehicle_person")).lower() == "true":
                counters["outside_person_violation_count"] += 1

            if evt == "PHONE" and role == "unknown":
                counters["unknown_role_phone_violation_count"] += 1

            if evt == "NO_SEATBELT" and str(context.get("motorcycle_flag")).lower() == "true":
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
    context_truth_path: Path | None = None,
    context_truth_lock_path: Path | None = None,
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

    if context_truth_path is None or context_truth_lock_path is None:
        raise ValueError(
            "frozen event evaluation requires separately frozen full-timeline context truth"
        )
    context_lock = json.loads(context_truth_lock_path.read_text(encoding="utf-8"))
    if context_lock.get("status") != "FROZEN_CONTEXT_GROUND_TRUTH":
        raise ValueError("context artifact is not FROZEN_CONTEXT_GROUND_TRUTH")
    if context_lock.get("coverage_status") != "FULL_TIMELINE_NO_GAPS_OR_OVERLAPS":
        raise ValueError("context artifact does not have complete timeline coverage")
    if context_lock.get("identity_status") != "KNOWN_STABLE_OCCUPANT_VEHICLE_CABIN":
        raise ValueError("context artifact does not have stable entity identity")
    if context_lock.get("human_review_status") != "ALL_APPROVED":
        raise ValueError("context artifact is not fully human-approved")
    if context_lock.get("adjudication_status") != "ALL_FINAL":
        raise ValueError("context artifact is not fully adjudicated")
    if context_lock.get("context_csv", {}).get("sha256") != sha256_file(context_truth_path):
        raise ValueError("context truth SHA256 does not match its frozen artifact")
    context_external = context_lock.get("external_test_lock", {})
    if context_external.get("sha256") != external_record.get("sha256"):
        raise ValueError("event and context truth do not bind the same external-test lock")
    context_external_path = Path(str(context_external.get("path", "")))
    if not context_external_path.is_file():
        raise ValueError("external-test artifact referenced by context truth is missing")
    if context_external.get("sha256") != sha256_file(context_external_path):
        raise ValueError("context truth external-test artifact SHA256 mismatch")
    if set(context_lock.get("video_ids", [])) != set(external_lock.get("video_ids", [])):
        raise ValueError("context truth does not cover every frozen external-test video")
    locked_minutes = float(context_lock.get("total_timeline_seconds", 0.0)) / 60.0
    if abs(locked_minutes - video_minutes) > 1e-6:
        raise ValueError(
            "--video-minutes must equal the duration proven by frozen context truth "
            f"({locked_minutes:.9f})"
        )

    truth_rows, _ = _read(truth_path)
    prediction_rows, pred_fields = _read(prediction_path)
    context_rows, _ = _read(context_truth_path, CONTEXT_REQUIRED)
    report = evaluate(
        truth_rows,
        prediction_rows,
        video_minutes,
        tolerance_seconds,
        prediction_fieldnames=pred_fields,
        context_rows=context_rows,
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
            "context_truth": {
                "path": str(context_truth_path),
                "sha256": sha256_file(context_truth_path),
                "lock_path": str(context_truth_lock_path),
                "lock_sha256": sha256_file(context_truth_lock_path),
                "coverage_status": context_lock.get("coverage_status"),
                "identity_status": context_lock.get("identity_status"),
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
        "context truth": report.get("context_truth", {}),
        "context-truth lock": {
            "path": report.get("context_truth", {}).get("lock_path"),
            "sha256": report.get("context_truth", {}).get("lock_sha256"),
        },
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
        "--context-truth",
        type=Path,
        help="Human-reviewed full-timeline context CSV",
    )
    parser.add_argument(
        "--context-truth-lock",
        type=Path,
        default=Path("datasets/manifests/v2_context_truth_frozen.json"),
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
        context_truth_path=args.context_truth,
        context_truth_lock_path=args.context_truth_lock,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
