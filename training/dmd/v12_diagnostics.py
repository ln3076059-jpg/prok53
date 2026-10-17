"""V12 read-only development diagnostics; never fits, tunes, or activates models."""

from __future__ import annotations

import json
import subprocess
import time
from collections import Counter
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from training.common import sha256_file
from training.dmd.behavior_roi import BehaviorRoiExtractor, Box, RoiStrategy
from training.dmd.causal_motion import (
    CausalMotionEncoder,
    causal_running_standardize,
    spatial_residual_standardize,
)
from training.dmd.online_candidate import CLASSES, OnlineE3
from training.dmd.research_permissions import (
    DEVELOPMENT,
    RECORD,
    read_permissions,
    require_operation,
    require_subject,
)

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v12_research"


def read(relative: str) -> dict:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def classification_metrics(labels: np.ndarray, predictions: np.ndarray) -> dict:
    if labels.ndim != 1 or predictions.shape != labels.shape or len(labels) == 0:
        raise ValueError("invalid classification arrays")
    if not np.isin(labels, [0, 1, 2]).all() or not np.isin(predictions, [0, 1, 2]).all():
        raise ValueError("invalid class index")
    matrix = np.zeros((3, 3), dtype=np.int64)
    np.add.at(matrix, (labels, predictions), 1)
    support, predicted = matrix.sum(axis=1), matrix.sum(axis=0)
    diagonal = matrix.diagonal()
    recall = np.divide(diagonal, support, out=np.zeros(3), where=support != 0)
    precision = np.divide(diagonal, predicted, out=np.zeros(3), where=predicted != 0)
    f1 = np.divide(
        2 * precision * recall, precision + recall, out=np.zeros(3), where=precision + recall != 0
    )
    return {
        "samples": len(labels),
        "confusion_matrix": matrix.tolist(),
        "macro_f1": round(float(f1.mean()), 4),
        "balanced_accuracy": round(float(recall[support > 0].mean()), 4),
        "safe_recall": round(float(recall[0]), 4),
        "phone_recall": round(float(recall[1]), 4),
        "other_recall": round(float(recall[2]), 4),
        "per_class": {
            name: {
                "precision": round(float(precision[i]), 4),
                "recall": round(float(recall[i]), 4),
                "f1": round(float(f1[i]), 4),
                "support": int(support[i]),
            }
            for i, name in enumerate(CLASSES)
        },
    }


def reconcile() -> dict:
    results = []
    sources = {}
    discrepancies = []
    for family in ("representation_experiments", "generalization_experiments"):
        source = f"reports/v11_research/{family}.json"
        sources[source] = sha256_file(ROOT / source)
        for representation in read(source)["representations"]:
            labels, predictions, folds = [], [], []
            for fold in representation["folds"]:
                require_subject(fold["heldout_subject"])
                path = ROOT / fold["prediction_path"]
                if path.resolve().parent != (ROOT / "reports/v11_research/predictions").resolve():
                    raise ValueError("unexpected prediction path")
                if sha256_file(path) != fold["prediction_sha256"]:
                    raise ValueError(f"prediction hash mismatch: {path}")
                sources[fold["prediction_path"]] = fold["prediction_sha256"]
                with np.load(path, allow_pickle=False) as saved:
                    y, pred = saved["labels"], saved["predictions"]
                    prob = saved["probabilities"]
                    if prob.shape != (len(y), 3) or not np.isfinite(prob).all():
                        raise ValueError("invalid prediction probabilities")
                    if np.any(prob < 0) or np.any(prob > 1):
                        raise ValueError("probability outside [0,1]")
                    np.testing.assert_allclose(prob.sum(axis=1), 1, atol=1e-5)
                    metrics = classification_metrics(y, pred)
                    for key, value in metrics.items():
                        if value != fold[key]:
                            discrepancies.append(f"{path.name}:{key}")
                    disagreements = int(np.sum(pred != prob.argmax(axis=1)))
                    timestamps = saved["timestamps"]
                    if np.any(np.diff(timestamps) <= 0):
                        raise ValueError("non-monotonic prediction timestamps")
                    folds.append(
                        {
                            "subject": fold["heldout_subject"],
                            "metrics": metrics,
                            "prediction_probability_argmax_disagreements": disagreements,
                            "timestamp_gaps_above_075s": int(np.sum(np.diff(timestamps) > 0.75)),
                        }
                    )
                    labels.append(y.copy())
                    predictions.append(pred.copy())
            pooled = classification_metrics(np.concatenate(labels), np.concatenate(predictions))
            if pooled != representation["pooled"]:
                discrepancies.append(representation["representation_id"] + ":pooled")
            results.append(
                {"id": representation["representation_id"], "folds": folds, "pooled": pooled}
            )
    return {
        "status": "PASS" if not discrepancies else "FAIL",
        "results": results,
        "discrepancies": discrepancies,
        "source_hashes": sources,
        "semantics": "Recomputed existing development predictions; no new model training",
    }


def streaming_audit() -> dict:
    require_operation("diagnostics")
    results = []
    for subject in sorted(DEVELOPMENT):
        require_subject(subject)
        path = ROOT / f"dataset/external_dmd/derivatives/v5_rgb_cache/v5_rgb_{subject}_64.npz"
        with np.load(path, allow_pickle=False) as data:
            frames = data["frames"]
            times = data["timestamps"]
            raw = CausalMotionEncoder().encode_sequence(frames)
            batch = causal_running_standardize(spatial_residual_standardize(raw))
            online = OnlineE3()
            actual, reasons, latencies, ready = [], Counter(), [], []
            for frame, timestamp in zip(frames, times, strict=True):
                begin = time.perf_counter()
                result = online.push(frame, float(timestamp), (subject, "BODY", "UNKNOWN"))
                latencies.append((time.perf_counter() - begin) * 1000)
                actual.append(result.feature)
                ready.append(result.ready)
                reasons.update(result.reasons)
            array = np.stack(actual)
            resets = reasons["CADENCE_GAP_RESET"]
            error = float(np.max(np.abs(array - batch)))
            parity = bool(np.allclose(array, batch, rtol=1e-4, atol=1e-4))
            if resets == 0 and not parity:
                raise ValueError(f"streaming parity failure: {subject}: {error}")
            np.savez_compressed(
                OUT / f"streaming_{subject}.npz",
                timestamps=times,
                features=array,
                ready=np.asarray(ready),
            )
            results.append(
                {
                    "subject": subject,
                    "source_path": path.relative_to(ROOT).as_posix(),
                    "source_sha256": sha256_file(path),
                    "samples": len(frames),
                    "max_absolute_error": error,
                    "parity": parity,
                    "reasons": dict(reasons),
                    "ready_samples": sum(ready),
                    "latency_ms": {
                        str(p): float(np.percentile(latencies, p)) for p in (50, 95, 99)
                    },
                    "scope": "64x64 cached RGB to E3 feature only; no decode/detector/head/fusion",
                    "state_max_frames": online.frames.maxlen,
                    "state_max_feature_rows": online.history.maxlen,
                }
            )
    return {
        "subjects": results,
        "training_performed": False,
        "e3_model_replay": "BLOCKED_MISSING_FITTED_HEAD_SCALER",
    }


def roi_audit() -> dict:
    require_operation("diagnostics")
    reviews = read("reports/v9_research/human_phone_review_summary.json")["cases"]
    mining = {
        r["case_id"]: r for r in read("reports/v9_research/phone_proposal_mining.json")["rows"]
    }
    manifest = read("reports/v11_research/dataset_manifest.json")
    captures = {}
    results = []
    source_hashes = {}
    try:
        for case in reviews:
            subject = case["subject_id"]
            require_subject(subject)
            folder = ROOT / "dataset/external_dmd/subjects" / subject
            paths = sorted(folder.glob("*_rgb_body.mp4"))
            if len(paths) != 1:
                raise ValueError(f"Expected single BODY video for {subject}")
            path = paths[0]
            if subject not in captures:
                source_hashes[subject] = sha256_file(path)
                expected = next(
                    v["sha256"]
                    for v in manifest["subjects"][subject]["views"]
                    if v["view"] == "body"
                )
                if source_hashes[subject] != expected:
                    raise ValueError("original BODY source hash mismatch")
                captures[subject] = cv2.VideoCapture(str(path))
            sample = case["selected_sample"] or mining[case["case_id"]]["samples"][0]
            capture = captures[subject]
            capture.set(cv2.CAP_PROP_POS_FRAMES, sample["frame_index"])
            ok, frame = capture.read()
            if not ok:
                raise ValueError(f"cannot decode original frame: {case['case_id']}")
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            driver = Box(*sample["driver_roi_xyxy_in_full_frame"])
            selected = case["selected_proposal"]
            phone = None
            if selected:
                x1, y1, x2, y2 = selected["xyxy_in_driver_roi"]
                phone = Box(x1 + driver.x1, y1 + driver.y1, x2 + driver.x1, y2 + driver.y1)
            crop = BehaviorRoiExtractor(output_size=(224, 224)).extract(
                rgb,
                driver,
                RoiStrategy.DYNAMIC_PADDED,
                source_camera="BODY",
                phone_box=phone,
                confidence=selected["confidence"] if selected else 0.0,
            )[0]
            roi_path = OUT / "roi_evidence" / f"{case['case_id']}.jpg"
            roi_path.parent.mkdir(exist_ok=True)
            if not cv2.imwrite(str(roi_path), cv2.cvtColor(crop.image, cv2.COLOR_RGB2BGR)):
                raise OSError("ROI image write failed")
            p = crop.provenance
            contained = (
                None
                if phone is None
                else (
                    p.crop_box[0] <= phone.x1 <= phone.x2 <= p.crop_box[2]
                    and p.crop_box[1] <= phone.y1 <= phone.y2 <= p.crop_box[3]
                )
            )
            results.append(
                {
                    "case_id": case["case_id"],
                    "subject": subject,
                    "original_video_path": path.relative_to(ROOT).as_posix(),
                    "original_video_sha256": source_hashes[subject],
                    "frame_index": sample["frame_index"],
                    "timestamp_seconds": sample["timestamp_seconds"],
                    "original_dimensions": [frame.shape[1], frame.shape[0]],
                    "phone_pixel_size_original": [phone.width, phone.height] if phone else None,
                    "phone_pixel_size_64_driver_cache": [
                        phone.width * 64 / driver.width,
                        phone.height * 64 / driver.height,
                    ]
                    if phone
                    else None,
                    "phone_contained_in_real_pixel_crop": contained,
                    "provenance": p.as_dict(),
                    "roi_path": roi_path.relative_to(ROOT).as_posix(),
                    "roi_sha256": sha256_file(roi_path),
                    "annotation_source": "EXISTING_AUTHENTICATED_HUMAN_REVIEW",
                    "scope": "Oracle reviewed-box crop diagnostic; not detector inference or AP",
                }
            )
    finally:
        for capture in captures.values():
            capture.release()
    return {
        "cases": results,
        "decoded_frames": len(results),
        "detector_rerun": False,
        "new_event_recovery_claimed": False,
    }


def stable_event_id(subject: str, kind: str, event: dict) -> str:
    # Stable across naive/aware comparisons. No confidence or GT-dependent prediction ID.
    return f"{subject}:{kind}:{event['start_seconds']:.6f}:{event['end_seconds']:.6f}"


def event_counts(matches: list, unmatched_preds: list, unmatched_gt: list) -> dict:
    unique_gt = {m["ground_truth"]["event_id"] for m in matches}
    duplicate_count = len(matches) - len(unique_gt)
    return {
        "tp": len(unique_gt),
        "fp": len(unmatched_preds) + duplicate_count,
        "fn": len(unmatched_gt),
        "duplicate_predictions": duplicate_count,
    }


def replay_audit() -> dict:
    require_operation("diagnostics")
    from backend.ai.events import Observation
    from training.dmd.adapter import parse_dmd_openlabel
    from training.dmd.temporal_eval import (
        TemporalMatchingPolicy,
        evaluate_temporal_engine_on_observations,
    )

    config = read("models/temporal_config_v2.json")
    frozen = read("reports/DMD_EVENT_MATCHING_POLICY_FREEZE.json")
    policy = TemporalMatchingPolicy(
        **{
            k: frozen[k]
            for k in ("iou_threshold", "onset_tolerance_seconds", "offset_tolerance_seconds")
        }
    )
    subjects = []
    sources = {}
    for subject, cache in (
        ("gC-14", "reports/cache_observations_sub_14.json"),
        ("gZ-36", "reports/cache_observations_sub_36.json"),
    ):
        require_subject(subject)
        data = read(cache)
        sources[cache] = sha256_file(ROOT / cache)
        annotations = sorted(
            (ROOT / "dataset/external_dmd/subjects" / subject).glob("*_ann_distraction.json")
        )
        if len(annotations) != 1:
            raise ValueError("expected single annotation")
        annotation = annotations[0]
        sources[annotation.relative_to(ROOT).as_posix()] = sha256_file(annotation)
        truth = [
            i.to_dict()
            for i in parse_dmd_openlabel(
                annotation,
                fps=data["diagnostics"]["fps"],
                total_frames=data["diagnostics"]["total_frames"],
            ).phone_intervals
        ]
        for interval in truth:
            interval["event_id"] = stable_event_id(subject, "GT", interval)
        observations = [Observation(**r) for r in data["observations"]]
        variants = {}
        for name in ("context_naive", "context_aware"):
            obs = (
                [
                    replace(r, phone_context="HANDHELD_USE") if r.confidence > 0 else r
                    for r in observations
                ]
                if name == "context_naive"
                else observations
            )
            metrics = evaluate_temporal_engine_on_observations(
                obs, truth, data["diagnostics"], config, policy
            )
            record = asdict(metrics)
            for match in record["matched_events"]:
                match["prediction"]["event_id"] = stable_event_id(
                    subject, "PRED", match["prediction"]
                )
            for pred in record["unmatched_predictions"]:
                pred["event_id"] = stable_event_id(subject, "PRED", pred)
            counts = event_counts(
                record["matched_events"], record["unmatched_predictions"], record["unmatched_gt"]
            )
            assert (counts["tp"], counts["fp"], counts["fn"]) == (
                metrics.true_positives,
                metrics.false_positives,
                metrics.false_negatives,
            )
            matched_ids = {m["ground_truth"]["event_id"] for m in record["matched_events"]}
            duration_bins = {}
            for lo, hi, label in (
                (0, 1, "lt1"),
                (1, 3, "1to3"),
                (3, 10, "3to10"),
                (10, float("inf"), "ge10"),
            ):
                group = [t for t in truth if lo <= t["end_seconds"] - t["start_seconds"] < hi]
                tp = sum(t["event_id"] in matched_ids for t in group)
                duration_bins[label] = {
                    "gt": len(group),
                    "tp": tp,
                    "recall": tp / len(group) if group else None,
                }
            record.update(
                recomputed_counts=counts,
                duration_bins=duration_bins,
                emission_latency="NOT_CAPTURED_BY_FROZEN_EVALUATOR",
            )
            variants[name] = record

        def tp_ids(name: str, variants: dict = variants) -> set[str]:
            return {m["ground_truth"]["event_id"] for m in variants[name]["matched_events"]}

        def fp_ids(name: str, variants: dict = variants) -> set[str]:
            seen, ids = set(), {r["event_id"] for r in variants[name]["unmatched_predictions"]}
            for match in variants[name]["matched_events"]:
                gt_id = match["ground_truth"]["event_id"]
                if gt_id in seen:
                    ids.add(match["prediction"]["event_id"])
                seen.add(gt_id)
            return ids

        subjects.append(
            {
                "subject": subject,
                "variants": variants,
                "delta": {
                    "recovered_tp_ids": sorted(tp_ids("context_aware") - tp_ids("context_naive")),
                    "lost_tp_ids": sorted(tp_ids("context_naive") - tp_ids("context_aware")),
                    "removed_fp_ids": sorted(fp_ids("context_naive") - fp_ids("context_aware")),
                    "new_fp_ids": sorted(fp_ids("context_aware") - fp_ids("context_naive")),
                },
            }
        )
    for path in ("models/temporal_config_v2.json", "reports/DMD_EVENT_MATCHING_POLICY_FREEZE.json"):
        sources[path] = sha256_file(ROOT / path)
    return {
        "scope": "FIXED_TWO_SUBJECT_HISTORICAL_CONTEXT_REPLAY_WITH_IDS",
        "subjects": subjects,
        "source_hashes": sources,
        "matcher": asdict(policy),
        "e3_replay": "BLOCKED_MISSING_FITTED_HEAD_SCALER_AND_VERIFIED_OCCUPANTS",
        "new_model_event_improvement_claimed": False,
    }


def run() -> None:
    OUT.mkdir(exist_ok=True)
    plan = OUT / "experiment_plan.json"
    if not plan.is_file():
        raise ValueError("record the experiment plan before execution")
    require_operation("diagnostics")
    start = datetime.now(UTC).isoformat()
    permission = read_permissions()
    denied = {}
    for operation in ("linear_head", "calibration", "fusion", "boundary", "backbone", "upload"):
        try:
            require_operation(operation)
        except PermissionError as exc:
            denied[operation] = str(exc)
    seatbelt_path = "dataset/manifests/v2_seatbelt_uncertain_ui_review.json"
    seatbelt = read(seatbelt_path)
    rows = seatbelt if isinstance(seatbelt, list) else seatbelt.get("selected", [])
    if not rows:
        raise ValueError("seatbelt manifest schema changed")
    from training.dmd.v10_audit import environment_audit, model_inventory

    inventory = model_inventory()
    write("model_manifest.json", inventory)
    prior = read("reports/v11_research/dataset_manifest.json")
    write(
        "dataset_manifest.json",
        {
            "permissions": permission,
            "permission_source": str(RECORD.relative_to(ROOT)),
            "permission_sha256": sha256_file(RECORD),
            "denied_operations": denied,
            "development_subjects": sorted(DEVELOPMENT),
            "independent_holdout": None,
            "protected_media_or_labels_opened": False,
            "subjects": prior["subjects"],
            "seatbelt_review": {
                "source": seatbelt_path,
                "sha256": sha256_file(ROOT / seatbelt_path),
                "samples": len(rows),
                "status_counts": dict(Counter(r.get("review_status") for r in rows)),
                "automatic_training_labels": sum(
                    r.get("automatic_training_label") is not None for r in rows
                ),
            },
        },
    )
    write(
        "repository_environment.json",
        {
            "head": subprocess.check_output(
                ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
            ).strip(),
            "environment": environment_audit(),
            "started_at_utc": start,
            "plan_sha256": sha256_file(plan),
        },
    )
    for name, function in (
        ("metric_reconciliation.json", reconcile),
        ("streaming_audit.json", streaming_audit),
        ("roi_audit.json", roi_audit),
        ("event_match_records.json", replay_audit),
    ):
        begin = time.perf_counter()
        result = function()
        result["elapsed_seconds"] = time.perf_counter() - begin
        write(name, result)
        print(name + " written", flush=True)
    write(
        "diagnostic_execution.json",
        {
            "started_at_utc": start,
            "finished_at_utc": datetime.now(UTC).isoformat(),
            "exit_code": 0,
            "training_performed": False,
            "production_approved": False,
        },
    )


if __name__ == "__main__":
    run()
