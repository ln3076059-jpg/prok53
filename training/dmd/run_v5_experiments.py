"""Governed Roadwatch V5 CPU research experiment.

Runs a leave-one-subject-out TinyCNN + temporal-pooling action branch on the three
development subjects, compares it with frozen V4 object evidence, and evaluates
conservative late fusion under the unchanged temporal matching policy.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import sha256_file, stable_json_hash
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.holdout_guard import assert_not_consumed_holdout_for_development
from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS
from training.dmd.v5_diagnostics import diagnose_v4_false_negatives
from training.dmd.v5_video_action import (
    ACTION_CLASSES,
    ClipDataset,
    SubjectFrames,
    TinyTemporalCNN,
    classification_metrics,
    event_metrics,
    extract_rgb_cache,
    merge_evidence_events,
    predict_subject,
    probabilities_to_events,
    train_model,
)

CONSUMED_HOLDOUTS = ("gZ-37", "gE-28")


def discover_external_dmd_root() -> Path:
    """Locate either the governed or explicitly restored local DMD tree."""
    candidates = (Path("datasets/external_dmd"), Path("dataset/external_dmd"))
    for candidate in candidates:
        if (candidate / "derivatives").is_dir() and (candidate / "subjects").is_dir():
            return candidate
    return candidates[0]


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _ground_truth(subjects_dir: Path, subject_id: str) -> list[dict[str, Any]]:
    annotation = parse_dmd_openlabel(_find_one(subjects_dir / subject_id, "*_ann_distraction.json"))
    return [item.to_dict() for item in annotation.phone_intervals]


def _pooled_event_metrics(per_subject: dict[str, dict[str, Any]]) -> dict[str, Any]:
    tp = sum(item["true_positives"] for item in per_subject.values())
    fp = sum(item["false_positives"] for item in per_subject.values())
    fn = sum(item["false_negatives"] for item in per_subject.values())
    minutes = sum(item["duration_seconds"] for item in per_subject.values()) / 60.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    latencies = [
        item["detection_latency_mean_seconds"]
        for item in per_subject.values()
        if item["detection_latency_mean_seconds"] is not None
    ]
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fp / max(minutes, 1e-9), 4),
        "detection_latency_mean_seconds": round(float(np.mean(latencies)), 3)
        if latencies
        else None,
        "duplicate_events": sum(item["duplicate_events"] for item in per_subject.values()),
        "event_fragmentation": sum(item["event_fragmentation"] for item in per_subject.values()),
        "evaluated_minutes": round(minutes, 3),
    }


def _evaluate_predictions(
    subjects_dir: Path,
    durations: dict[str, float],
    predictions: dict[str, list[dict[str, Any]]],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    per_subject: dict[str, dict[str, Any]] = {}
    for subject_id in DEVELOPMENT_SUBJECTS:
        result = event_metrics(
            _ground_truth(subjects_dir, subject_id),
            predictions[subject_id],
            durations[subject_id],
        )
        result["duration_seconds"] = durations[subject_id]
        per_subject[subject_id] = result
    return _pooled_event_metrics(per_subject), per_subject


def _calibrate_threshold(
    config: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    training_subjects: tuple[str, ...],
    subjects_dir: Path,
    durations: dict[str, float],
) -> tuple[float, list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    event_cfg = config["event_decision"]
    for threshold in event_cfg["action_threshold_grid"]:
        per_subject: dict[str, dict[str, Any]] = {}
        for subject_id in training_subjects:
            prediction = predictions[subject_id]
            events = probabilities_to_events(
                prediction["timestamps"],
                prediction["probabilities"],
                threshold=float(threshold),
                release_threshold=float(threshold) * float(event_cfg["release_ratio"]),
                minimum_samples=int(event_cfg["minimum_positive_samples"]),
                merge_gap_seconds=float(event_cfg["merge_gap_seconds"]),
            )
            metrics = event_metrics(
                _ground_truth(subjects_dir, subject_id), events, durations[subject_id]
            )
            metrics["duration_seconds"] = durations[subject_id]
            per_subject[subject_id] = metrics
        pooled = _pooled_event_metrics(per_subject)
        rows.append({"threshold": float(threshold), **pooled})
    eligible = [row for row in rows if row["false_alarms_per_minute"] <= 0.30]
    candidates = eligible or rows
    selected = max(
        candidates,
        key=lambda row: (
            row["f1"],
            row["recall"],
            -row["false_alarms_per_minute"],
            row["threshold"],
        ),
    )
    return float(selected["threshold"]), rows


def _serializable_prediction(prediction: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in prediction.items()
        if key not in {"centers", "timestamps", "labels", "probabilities"}
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = discover_external_dmd_root()
    derivatives_dir = data_root / "derivatives"
    subjects_dir = data_root / "subjects"
    v4_path = Path("reports/v4_upgrade/v4_development_results.json")
    v4 = json.loads(v4_path.read_text(encoding="utf-8"))
    selected_v4 = next(
        item for item in v4["experiments"] if item["experiment_id"] == v4["selected_candidate"]
    )
    action_cfg = config["video_action"]
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir = derivatives_dir / "v5_rgb_cache"
    checkpoint_dir = Path("models/candidate/v5_tiny_temporal_cnn_001")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    blockers: list[str] = []
    inputs: list[dict[str, Any]] = []
    subjects: dict[str, SubjectFrames] = {}
    durations: dict[str, float] = {}
    for subject_id in DEVELOPMENT_SUBJECTS:
        assert_not_consumed_holdout_for_development(subject_id, caller_action="v5_loso")
        subject_dir = subjects_dir / subject_id
        feature_path = derivatives_dir / f"v3_features_{subject_id}.json"
        if not feature_path.is_file():
            blockers.append(f"missing V3 feature alignment cache: {feature_path}")
            continue
        feature_payload = json.loads(feature_path.read_text(encoding="utf-8"))
        video_path = _find_one(subject_dir, "*_rgb_body.mp4")
        annotation_path = _find_one(subject_dir, "*_ann_distraction.json")
        cache_path = cache_dir / f"v5_rgb_{subject_id}_{int(action_cfg['image_size'])}.npz"
        if not cache_path.is_file():
            cache_metadata = extract_rgb_cache(
                subject_id,
                video_path,
                feature_payload,
                annotation_path,
                cache_path,
                roi_xyxy=tuple(float(value) for value in action_cfg["driver_roi_xyxy"]),
                image_size=int(action_cfg["image_size"]),
            )
        else:
            cache_metadata = {
                "subject_id": subject_id,
                "status": "REUSED",
                "output_path": str(cache_path),
                "output_sha256": sha256_file(cache_path),
                "video_path": str(video_path),
                "video_sha256": sha256_file(video_path),
                "annotation_path": str(annotation_path),
                "annotation_sha256": sha256_file(annotation_path),
            }
        inputs.append(cache_metadata)
        subjects[subject_id] = SubjectFrames.load(subject_id, cache_path)
        durations[subject_id] = float(feature_payload["duration_seconds"])

    if blockers:
        result = {
            "status": "BLOCKED",
            "blockers": blockers,
            "development_subjects": list(DEVELOPMENT_SUBJECTS),
            "forbidden_consumed_holdouts": list(CONSUMED_HOLDOUTS),
        }
        (output_dir / "v5_experiment_results.json").write_text(
            json.dumps(result, indent=2), encoding="utf-8"
        )
        return result

    folds: list[dict[str, Any]] = []
    heldout_predictions: dict[str, dict[str, Any]] = {}
    for fold_index, heldout_subject in enumerate(DEVELOPMENT_SUBJECTS):
        training_subjects = tuple(item for item in DEVELOPMENT_SUBJECTS if item != heldout_subject)
        checkpoint_path = checkpoint_dir / f"loso_{heldout_subject}.pt"
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (
                checkpoint.get("heldout_subject") != heldout_subject
                or tuple(checkpoint.get("training_subjects", ())) != training_subjects
                or checkpoint.get("architecture") != action_cfg["architecture"]
            ):
                raise ValueError(f"checkpoint provenance mismatch: {checkpoint_path}")
            model = TinyTemporalCNN()
            model.load_state_dict(checkpoint["model_state_dict"])
            training = {
                "status": "REUSED_COMPLETED_CHECKPOINT",
                "epochs": int(action_cfg["epochs"]),
                "training_seconds": None,
            }
        else:
            train_dataset = ClipDataset(
                [subjects[item] for item in training_subjects],
                sequence_length=int(action_cfg["sequence_length"]),
                stride=int(action_cfg["training_stride"]),
                labeled_only=True,
            )
            model, training = train_model(
                train_dataset,
                epochs=int(action_cfg["epochs"]),
                batch_size=int(action_cfg["batch_size"]),
                learning_rate=float(action_cfg["learning_rate"]),
                seed=int(action_cfg["seed"]) + fold_index,
            )
        train_predictions = {
            item: predict_subject(
                model,
                subjects[item],
                sequence_length=int(action_cfg["sequence_length"]),
                batch_size=int(action_cfg["batch_size"]),
            )
            for item in training_subjects
        }
        threshold, threshold_sweep = _calibrate_threshold(
            config, train_predictions, training_subjects, subjects_dir, durations
        )
        heldout = predict_subject(
            model,
            subjects[heldout_subject],
            sequence_length=int(action_cfg["sequence_length"]),
            batch_size=int(action_cfg["batch_size"]),
        )
        heldout_predictions[heldout_subject] = heldout
        heldout_events = probabilities_to_events(
            heldout["timestamps"],
            heldout["probabilities"],
            threshold=threshold,
            release_threshold=threshold * float(config["event_decision"]["release_ratio"]),
            minimum_samples=int(config["event_decision"]["minimum_positive_samples"]),
            merge_gap_seconds=float(config["event_decision"]["merge_gap_seconds"]),
        )
        heldout["events"] = heldout_events
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "architecture": action_cfg["architecture"],
            "class_names": ACTION_CLASSES,
            "heldout_subject": heldout_subject,
            "training_subjects": training_subjects,
            "action_threshold": threshold,
            "config": action_cfg,
            "dataset_role": "DEVELOPMENT_ONLY_LOSO",
            "forbidden_consumed_holdouts": CONSUMED_HOLDOUTS,
            "git_head": _git_head(),
        }
        torch.save(checkpoint_payload, checkpoint_path)
        fold_event = event_metrics(
            _ground_truth(subjects_dir, heldout_subject), heldout_events, durations[heldout_subject]
        )
        fold_event["duration_seconds"] = durations[heldout_subject]
        folds.append(
            {
                "fold": fold_index + 1,
                "training_subjects": list(training_subjects),
                "heldout_subject": heldout_subject,
                "training": training,
                "selected_action_threshold": threshold,
                "threshold_sweep_training_subjects_only": threshold_sweep,
                "heldout_clip_metrics": classification_metrics(
                    heldout["labels"], heldout["probabilities"]
                ),
                "heldout_event_metrics": fold_event,
                "runtime": _serializable_prediction(heldout),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )

    pooled_clip = classification_metrics(
        np.concatenate([heldout_predictions[item]["labels"] for item in DEVELOPMENT_SUBJECTS]),
        np.concatenate(
            [heldout_predictions[item]["probabilities"] for item in DEVELOPMENT_SUBJECTS], axis=0
        ),
    )
    action_events = {
        subject_id: heldout_predictions[subject_id]["events"] for subject_id in DEVELOPMENT_SUBJECTS
    }
    action_pooled, action_per_subject = _evaluate_predictions(
        subjects_dir, durations, action_events
    )
    object_events = {
        subject_id: selected_v4["per_subject"][subject_id]["evaluation_evidence"][
            "predicted_events"
        ]
        for subject_id in DEVELOPMENT_SUBJECTS
    }
    fused_events = {
        subject_id: merge_evidence_events(object_events[subject_id], action_events[subject_id])
        for subject_id in DEVELOPMENT_SUBJECTS
    }
    fused_pooled, fused_per_subject = _evaluate_predictions(
        subjects_dir, durations, fused_events
    )
    reliable_events = {
        subject_id: merge_evidence_events(
            object_events[subject_id],
            [
                item
                for item in action_events[subject_id]
                if item["confidence"] >= 0.70 and item["duration"] >= 1.0
            ],
        )
        for subject_id in DEVELOPMENT_SUBJECTS
    }
    reliable_pooled, reliable_per_subject = _evaluate_predictions(
        subjects_dir, durations, reliable_events
    )

    experiments = [
        {
            "experiment_id": "B0_V4_CURRENT",
            "name": "Frozen V4 object/geometry pipeline",
            "pooled_metrics": {
                "precision": selected_v4["pooled_metrics"]["precision"] / 100.0,
                "recall": selected_v4["pooled_metrics"]["recall"] / 100.0,
                "f1": selected_v4["pooled_metrics"]["f1"] / 100.0,
                "false_alarms_per_minute": selected_v4["pooled_metrics"][
                    "false_alarms_per_minute"
                ],
                "true_positives": selected_v4["pooled_metrics"]["true_positives"],
                "false_positives": selected_v4["pooled_metrics"]["false_positives"],
                "false_negatives": selected_v4["pooled_metrics"]["false_negatives"],
                "evaluated_minutes": selected_v4["pooled_metrics"]["evaluated_minutes"],
            },
            "per_subject": selected_v4["per_subject"],
            "status": "MEASURED_ON_ROADWATCH",
        },
        {
            "experiment_id": "B2_TINY_CNN_TEMPORAL",
            "name": "Independent BODY RGB TinyCNN with temporal pooling",
            "pooled_metrics": action_pooled,
            "per_subject": action_per_subject,
            "status": "MEASURED_ON_ROADWATCH_SUBJECT_DISJOINT",
        },
        {
            "experiment_id": "B5_LATE_FUSION",
            "name": "Object events plus action NEEDS_REVIEW candidates",
            "pooled_metrics": fused_pooled,
            "per_subject": fused_per_subject,
            "status": "MEASURED_ON_ROADWATCH_SUBJECT_DISJOINT",
        },
        {
            "experiment_id": "B6_RELIABILITY_TEMPORAL",
            "name": "Reliability-gated action candidates plus object events",
            "pooled_metrics": reliable_pooled,
            "per_subject": reliable_per_subject,
            "status": "MEASURED_ON_ROADWATCH_SUBJECT_DISJOINT",
        },
    ]
    eligible = [
        item
        for item in experiments
        if item["pooled_metrics"]["false_alarms_per_minute"]
        <= float(config["selection"]["maximum_false_alarms_per_minute"])
    ]
    selected = max(
        eligible,
        key=lambda item: (
            item["pooled_metrics"]["f1"],
            item["pooled_metrics"]["recall"],
            -item["pooled_metrics"]["false_alarms_per_minute"],
        ),
    )

    diagnostics = diagnose_v4_false_negatives(derivatives_dir, subjects_dir, v4_path)
    selected_per_subject = selected["per_subject"]
    recovered: list[dict[str, Any]] = []
    for case in diagnostics["false_negative_cases"]:
        subject_id = case["subject_id"]
        truth = case["truth"]
        for match in selected_per_subject[subject_id].get("matches", []):
            candidate = match["ground_truth"]
            if (
                float(candidate["start_seconds"]) == float(truth["start_seconds"])
                and float(candidate["end_seconds"]) == float(truth["end_seconds"])
            ):
                recovered.append(
                    {
                        "case_id": case["case_id"],
                        "subject_id": subject_id,
                        "source_action": truth["source_action"],
                        "v4_failure_stage": case["primary_failure_stage"],
                        "selected_v5_prediction": match["prediction"],
                        "matching_evidence": {
                            "iou": match["iou"],
                            "start_latency": match["start_latency"],
                            "end_latency": match["end_latency"],
                        },
                    }
                )
                break

    runtime_rows = [fold["runtime"] for fold in folds]
    checkpoint_metadata_path = checkpoint_dir / "metadata.json"
    checkpoint_metadata = {
        "schema_version": 1,
        "model_version": config["model_version"],
        "architecture": action_cfg["architecture"],
        "activation_state": config["activation_state"],
        "dataset_role": config["dataset_role"],
        "action_classes": list(ACTION_CLASSES),
        "evaluation_protocol": "LEAVE_ONE_SUBJECT_OUT",
        "forbidden_consumed_holdouts": list(CONSUMED_HOLDOUTS),
        "automatic_violation": False,
        "folds": [
            {
                "heldout_subject": fold["heldout_subject"],
                "training_subjects": fold["training_subjects"],
                "action_threshold": fold["selected_action_threshold"],
                "checkpoint_path": fold["checkpoint_path"],
                "checkpoint_sha256": fold["checkpoint_sha256"],
            }
            for fold in folds
        ],
    }
    checkpoint_metadata_path.write_text(
        json.dumps(checkpoint_metadata, indent=2), encoding="utf-8"
    )
    result = {
        "status": "COMPLETED",
        "created_at": datetime.now(UTC).isoformat(),
        "git_head": _git_head(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_role": "DEVELOPMENT_ONLY",
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "forbidden_consumed_holdouts": list(CONSUMED_HOLDOUTS),
        "holdouts_accessed": [],
        "evaluation_protocol": "LEAVE_ONE_SUBJECT_OUT; threshold calibration uses training subjects only",
        "input_provenance": inputs,
        "input_set_sha256": stable_json_hash(inputs),
        "action_classes": list(ACTION_CLASSES) + ["UNKNOWN (abstention only; not ground truth)"],
        "clip_metrics": pooled_clip,
        "folds": folds,
        "checkpoint_metadata_path": str(checkpoint_metadata_path),
        "checkpoint_metadata_sha256": sha256_file(checkpoint_metadata_path),
        "experiments": experiments,
        "selection_rule": (
            "Highest pooled event F1 with FA/min <= 0.30, then recall, then lower FA/min"
        ),
        "selected_candidate": selected["experiment_id"],
        "selected_metrics": selected["pooled_metrics"],
        "v4_false_negatives_recovered": recovered,
        "recovered_count": len(recovered),
        "runtime": {
            "device": "CPU",
            "torch_version": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "python": platform.python_version(),
            "platform": platform.platform(),
            "per_fold": runtime_rows,
            "mean_clips_per_second": round(
                float(np.mean([row["clips_per_second"] for row in runtime_rows])), 3
            ),
            "p50_latency_ms_across_folds": round(
                float(np.median([row["p50_latency_ms"] for row in runtime_rows])), 3
            ),
            "p95_latency_ms_across_folds": round(
                float(np.max([row["p95_latency_ms"] for row in runtime_rows])), 3
            ),
            "peak_process_rss_mb": round(
                float(np.max([row["peak_process_rss_mb"] for row in runtime_rows])), 3
            ),
            "ram_vram_note": (
                "Process RSS was sampled every 10 ms during inference; CUDA is unavailable, "
                "so VRAM is not applicable. RSS includes the Python/PyTorch process, model, "
                "and loaded subject cache, not only model weights."
            ),
        },
        "governance": {
            "action_only_event_status": "NEEDS_REVIEW",
            "automatic_violation": False,
            "generalization_claim": "NOT_ESTABLISHED",
            "independent_holdout": "NOT_AVAILABLE",
        },
    }
    (output_dir / "04_false_negative_diagnostics.json").write_text(
        json.dumps(diagnostics, indent=2), encoding="utf-8"
    )
    (output_dir / "v5_experiment_results.json").write_text(
        json.dumps(result, indent=2), encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Roadwatch V5 CPU research experiment")
    parser.add_argument(
        "--config", type=Path, default=Path("models/model_config_v5_research.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/v5_research_upgrade")
    )
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    print(json.dumps(result, indent=2))
    if result["status"] == "BLOCKED":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
