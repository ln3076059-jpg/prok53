"""Recompute the latest comparator and trace every development event and error."""

from __future__ import annotations

import hashlib
import json
import platform
import shutil
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch

from training.common import sha256_file
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS
from training.dmd.run_v5_experiments import (
    _evaluate_predictions,
    _ground_truth,
    discover_external_dmd_root,
)
from training.dmd.v5_video_action import ACTION_CLASSES, label_timestamps


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def artifact(path: Path) -> dict[str, Any]:
    return {"path": path.as_posix(), "exists": path.is_file(), **({
        "bytes": path.stat().st_size, "sha256": sha256_file(path),
    } if path.is_file() else {})}


def interval_key(event: dict[str, Any]) -> tuple[float, float]:
    return float(event["start_seconds"]), float(event["end_seconds"])


def baseline_inputs() -> tuple[dict[str, Any], dict[str, Any], dict[str, float]]:
    result = read_json("reports/v7_optimization/v7_experiments.json")
    baseline = result["variants"][0]
    root = discover_external_dmd_root()
    durations = {
        s: float(read_json(root / "derivatives" / f"v3_features_{s}.json")["duration_seconds"])
        for s in DEVELOPMENT_SUBJECTS
    }
    predictions = {}
    for subject in DEVELOPMENT_SUBJECTS:
        path = Path("reports/v6_completion/folds") / f"hierarchical_loss_only_loso_{subject}"
        with np.load(path.with_suffix(".npz"), allow_pickle=False) as payload:
            predictions[subject] = {key: payload[key].copy() for key in payload.files}
        predictions[subject]["events"] = read_json(path.with_suffix(".json"))["events"]
    return baseline, predictions, durations


def _case_evidence(
    event: dict[str, Any], records: list[dict[str, Any]], prediction: dict[str, Any]
) -> dict[str, Any]:
    start, end = interval_key(event)
    rows = [r for r in records if start <= float(r["timestamp_seconds"]) <= end]
    timestamps, p = prediction["timestamps"], prediction["probabilities"]
    mask = (timestamps >= start) & (timestamps <= end)
    phone = p[:, 1:5].sum(axis=1)
    smooth = np.convolve(phone, np.ones(3) / 3, mode="same")
    local = phone[mask]
    labels = prediction["labels"][mask]
    raw = {
        view: max((float(r[f"{view}_phone_conf"]) for r in rows), default=0.0)
        for view in ("body", "face", "hands")
    }
    strong = smooth[mask] >= 0.8
    return {
        "max_detector_confidence": raw,
        "sample_count": len(local),
        "phone_probability_max": float(local.max()) if len(local) else None,
        "phone_probability_mean": float(local.mean()) if len(local) else None,
        "smoothed_phone_probability_max": float(smooth[mask].max()) if len(local) else None,
        "consecutive_activation_samples": bool(np.any(strong[:-1] & strong[1:])),
        "predicted_class_counts": dict(Counter(ACTION_CLASSES[i] for i in p[mask].argmax(1))),
        "ground_truth_class_counts": dict(Counter(
            ACTION_CLASSES[i] if i >= 0 else "UNKNOWN" for i in labels
        )),
        "driver_phone_association": "UNAVAILABLE_IN_LEGACY_CONFIDENCE_ONLY_CACHE",
        "geometry_note": "Pose vectors are recorded; boxes and per-phone identities are absent.",
        "trace": [{
            "timestamp_seconds": float(t), "class_probabilities": probs.tolist(),
            "smoothed_phone_probability": float(score),
        } for t, probs, score in zip(timestamps[mask], p[mask], smooth[mask], strict=True)],
        "detector_pose_trace": rows,
    }


def audit_errors(output_dir: Path) -> dict[str, Any]:
    baseline, predictions, durations = baseline_inputs()
    root = discover_external_dmd_root()
    fused = baseline["experiments"][1]
    pooled, per_subject = _evaluate_predictions(
        root / "subjects", durations,
        {s: fused["per_subject"][s]["predicted_events"] for s in DEVELOPMENT_SUBJECTS},
    )
    if pooled != fused["pooled_metrics"]:
        raise ValueError("latest baseline does not reproduce under the frozen evaluator")
    cases, false_positives = [], []
    reached = Counter()
    for subject in DEVELOPMENT_SUBJECTS:
        records = read_json(root / "derivatives" / f"v3_features_{subject}.json")["records"]
        evaluation = per_subject[subject]
        matched = {interval_key(m["ground_truth"]) for m in evaluation["matches"]}
        for index, truth in enumerate(_ground_truth(root / "subjects", subject)):
            evidence = _case_evidence(truth, records, predictions[subject])
            detected = max(evidence["max_detector_confidence"].values()) > 0
            candidate = max(evidence["max_detector_confidence"].values()) >= 0.10
            activated = evidence["consecutive_activation_samples"]
            action_overlap = [e for e in predictions[subject]["events"] if
                              min(e["end_seconds"], truth["end_seconds"]) >
                              max(e["start_seconds"], truth["start_seconds"])]
            checks = {"any_detector_response": detected, "detector_confidence_ge_010": candidate,
                      "action_sustained_ge_080": activated,
                      "action_event_overlap": bool(action_overlap),
                      "fused_event_match": interval_key(truth) in matched}
            reached.update(k for k, v in checks.items() if v)
            is_fn = not checks["fused_event_match"]
            reason = "MATCHED" if not is_fn else (
                "ACTION_PROBABILITY_OR_SUSTAINED_ACTIVATION" if not activated else
                "TEMPORAL_BOUNDARY_OR_EVENT_MATCHING"
            )
            cases.append({"case_id": f"{subject}-phone-{index + 1:02d}",
                          "subject_id": subject, "truth": truth,
                          "result": "FN" if is_fn else "TP", "checks": checks,
                          "action_branch_failure": reason, "evidence": evidence})
        seen = set()
        for match in evaluation["matches"]:
            key = interval_key(match["ground_truth"])
            if key in seen:
                false_positives.append({"subject_id": subject, "kind": "DUPLICATE_MATCH",
                                        "event": match["prediction"], "match": match,
                                        "evidence": _case_evidence(
                                            match["prediction"], records, predictions[subject])})
            seen.add(key)
        for event in evaluation["unmatched_predictions"]:
            false_positives.append({"subject_id": subject, "kind": "UNMATCHED_PREDICTION",
                                    "event": event, "evidence": _case_evidence(
                                        event, records, predictions[subject])})
    if len(false_positives) != pooled["false_positives"]:
        raise ValueError("FP audit must account for duplicates as well as unmatched events")
    result = {
        "baseline_id": fused["experiment_id"], "recomputed_metrics": pooled,
        "per_subject": per_subject, "coarse_metrics": baseline["coarse_metrics"],
        "clip_metrics": baseline["clip_metrics"], "total_ground_truth_events": len(cases),
        "stage_coverage": {k: {"events": n, "fraction": n / len(cases)}
                           for k, n in reached.items()},
        "stage_coverage_note": "Parallel branch coverage, not sequential conditional recall.",
        "cases": cases, "false_positive_cases": false_positives,
        "unknowns": ["Object association and visual occlusion cannot be inferred from confidences.",
                     "Legacy fusion scores and temporal internal state are not in this cache.",
                     "Stage coverage does not establish causal loss without a replay ablation."],
    }
    write_json(output_dir / "error_analysis.json", result)
    return result


def audit_dataset(output_dir: Path) -> dict[str, Any]:
    root = discover_external_dmd_root()
    subjects, fingerprints, cross_duplicates = {}, {}, []
    assets = []
    for subject in DEVELOPMENT_SUBJECTS:
        cache = root / "derivatives/v5_resnet18_embeddings" / f"v5_resnet18_{subject}.npz"
        rgb_path = root / "derivatives/v5_rgb_cache" / f"v5_rgb_{subject}_64.npz"
        annotation = next((root / "subjects" / subject).glob("*_ann_distraction.json"))
        with np.load(cache, allow_pickle=False) as payload:
            embeddings = payload["embeddings"].astype(np.float32)
            timestamps, labels = payload["timestamps"], payload["labels"]
        with np.load(rgb_path, allow_pickle=False) as rgb:
            frames = rgb["frames"]
            source_timestamps = rgb["timestamps"]
            aligned = bool(np.array_equal(timestamps, rgb["timestamps"]) and
                           np.array_equal(labels, rgb["labels"]))
            indices = rgb["frame_indices"]
        parsed = parse_dmd_openlabel(annotation)
        expected, _ = label_timestamps(source_timestamps, parsed)
        widened, _ = label_timestamps(timestamps, parsed)
        hashes = [hashlib.sha256(frame.tobytes()).hexdigest() for frame in frames]
        for index, digest in enumerate(hashes):
            previous = fingerprints.get(digest)
            if previous and previous[0] != subject:
                cross_duplicates.append({"first": previous, "second": [subject, index]})
            fingerprints[digest] = (subject, index)
        subjects[subject] = {
            "samples": len(labels), "class_counts": dict(Counter(
                ACTION_CLASSES[i] if i >= 0 else "UNKNOWN" for i in labels)),
            "rgb_embedding_aligned": aligned,
            "annotation_label_mismatches": int(np.sum(labels != expected)),
            "float64_boundary_replay_disagreements": int(np.sum(labels != widened)),
            "label_replay_dtype": str(source_timestamps.dtype),
            "strictly_increasing_timestamps": bool(np.all(np.diff(timestamps) > 0)),
            "sample_interval_median_seconds": float(np.median(np.diff(timestamps))),
            "timestamp_frame_residual_max_seconds": float(np.max(
                np.abs(timestamps - indices / 29.76))),
            "embedding_finite": bool(np.isfinite(embeddings).all()),
            "embedding_dim": embeddings.shape[1], "rgb_shape": list(frames.shape),
            "exact_duplicate_rgb_frames_within_subject": len(hashes) - len(set(hashes)),
            "embedding_temporal_difference_mean_l2": float(np.linalg.norm(
                np.diff(embeddings, axis=0), axis=1).mean()),
            "embedding_subject_mean": embeddings.mean(axis=0).tolist(),
            "available_views": [v for v in ("body", "face", "hands")
                                if list((root / "subjects" / subject).glob(f"*_rgb_{v}.mp4"))],
        }
        assets.extend(artifact(p) for p in (cache, rgb_path, annotation))
    mean_distances = []
    for i, first in enumerate(DEVELOPMENT_SUBJECTS):
        for second in DEVELOPMENT_SUBJECTS[i + 1:]:
            mean_distances.append({"subjects": [first, second], "mean_l2_distance": float(
                np.linalg.norm(np.asarray(subjects[first]["embedding_subject_mean"]) -
                               np.asarray(subjects[second]["embedding_subject_mean"]))
            )})
    for row in subjects.values():
        row.pop("embedding_subject_mean")
    result = {
        "development_subjects": list(DEVELOPMENT_SUBJECTS), "subjects": subjects,
        "cross_subject_exact_rgb_duplicates": cross_duplicates,
        "subject_embedding_mean_distances": mean_distances, "assets": assets,
        "consumed_holdouts_read": [], "new_independent_holdout": None,
        "near_duplicate_status": "NOT_MEASURED_NO_CLAIM_OF_ABSENCE",
        "lighting_occlusion_distribution": "NO_VERIFIED_CASE_LEVEL_LABELS",
        "license": "Inherited DMD source; no new acquisition or redistribution.",
        "provenance": "Existing DMD annotations; no model-generated human approvals.",
    }
    write_json(output_dir / "dataset_audit.json", result)
    return result


def audit_environment(output_dir: Path) -> dict[str, Any]:
    status = subprocess.check_output(["git", "status", "--porcelain"], text=True).splitlines()
    remote = subprocess.run(
        ["git", "ls-remote", "origin", "HEAD", "refs/heads/main",
         "refs/heads/roadwatch-v4-candidate"], capture_output=True, text=True, timeout=30,
    )
    result = {
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], text=True).strip(),
        "status_counts": dict(Counter(line[:2] for line in status)),
        "modified_source_paths": [line[3:] for line in status if line.startswith(" M")],
        "remote_exit_code": remote.returncode, "remote_refs": remote.stdout.splitlines(),
        "platform": platform.platform(), "python": platform.python_version(),
        "torch": torch.__version__, "cuda_available": torch.cuda.is_available(),
        "memory": psutil.virtual_memory()._asdict(), "disk": shutil.disk_usage(".")._asdict(),
        "active_models": [artifact(p) for p in sorted(Path("models/active/v2").glob("*.pt"))],
        "working_tree_preserved": True,
    }
    write_json(output_dir / "environment_audit.json", result)
    return result


def export_fn_frames(output_dir: Path, analysis: dict[str, Any]) -> list[dict[str, Any]]:
    """Export original BODY ROIs for review, without creating or revising labels."""
    root = discover_external_dmd_root()
    image_dir = output_dir / "fn_original_rois"
    image_dir.mkdir(exist_ok=True, parents=True)
    exported = []
    for subject in DEVELOPMENT_SUBJECTS:
        records = read_json(root / "derivatives" / f"v3_features_{subject}.json")["records"]
        video = next((root / "subjects" / subject).glob("*_rgb_body.mp4"))
        cap = cv2.VideoCapture(str(video))
        try:
            for case in analysis["cases"]:
                if case["subject_id"] != subject or case["result"] != "FN":
                    continue
                start, end = interval_key(case["truth"])
                record = min(records, key=lambda r: abs(r["timestamp_seconds"] - (start + end) / 2))
                cap.set(cv2.CAP_PROP_POS_FRAMES, record["frame_index"])
                ok, frame = cap.read()
                if not ok:
                    exported.append({"case_id": case["case_id"], "status": "DECODE_FAILED"})
                    continue
                h, w = frame.shape[:2]
                roi = frame[int(0.10 * h):int(0.95 * h), :int(0.65 * w)]
                target = image_dir / f"{case['case_id']}.jpg"
                if not cv2.imwrite(str(target), roi):
                    raise OSError(f"cannot save ROI: {target}")
                exported.append({"case_id": case["case_id"], "status": "EXPORTED",
                                 "timestamp_seconds": record["timestamp_seconds"],
                                 "frame_index": record["frame_index"],
                                 "shape": list(roi.shape), **artifact(target)})
        finally:
            cap.release()
    write_json(output_dir / "fn_frame_manifest.json", exported)
    return exported
