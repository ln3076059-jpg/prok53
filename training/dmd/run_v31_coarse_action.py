"""Preregistered three-class causal action experiment on V31 direct112 embeddings."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from training.common import sha256_file, stable_json_hash
from training.dmd.research_permissions import require_operation
from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS
from training.dmd.run_v5_experiments import _ground_truth, discover_external_dmd_root
from training.dmd.v30_causal_action import (
    ActionDecoderConfig,
    CausalEmbeddingDataset,
    predict_causal_action,
    train_causal_action_model,
)
from training.dmd.v5_resnet_gru import SubjectEmbeddings
from training.dmd.v5_video_action import event_metrics

COARSE_CLASSES = ("SAFE_DRIVING", "PHONE_USE", "OTHER_ACTIVITY")
PHONE_CLASS = 1


def remap_coarse_labels(labels: np.ndarray) -> np.ndarray:
    """Map six-class DMD targets to SAFE/PHONE/OTHER while preserving UNKNOWN."""

    mapped = np.full(np.asarray(labels).shape, -1, dtype=np.int64)
    mapped[np.asarray(labels) == 0] = 0
    mapped[np.isin(labels, (1, 2, 3, 4))] = 1
    mapped[np.asarray(labels) == 5] = 2
    return mapped


def coarse_subject(subject: SubjectEmbeddings) -> SubjectEmbeddings:
    return SubjectEmbeddings(
        subject_id=subject.subject_id,
        embeddings=subject.embeddings,
        timestamps=subject.timestamps,
        labels=remap_coarse_labels(subject.labels),
    )


def coarse_classification_metrics(
    labels: np.ndarray, probabilities: np.ndarray
) -> dict[str, Any]:
    valid = labels >= 0
    truth = labels[valid]
    predicted = probabilities[valid].argmax(axis=1) if np.any(valid) else np.array([], dtype=int)
    confusion = np.zeros((len(COARSE_CLASSES), len(COARSE_CLASSES)), dtype=np.int64)
    for actual, guess in zip(truth, predicted, strict=True):
        confusion[int(actual), int(guess)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(COARSE_CLASSES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "support": int(confusion[index].sum()),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return {
        "clip_accuracy": round(float(np.mean(truth == predicted)), 4) if len(truth) else 0.0,
        "macro_f1": round(float(np.mean(f1_values)), 4),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_order": list(COARSE_CLASSES),
        "evaluated_clips": int(len(truth)),
        "unknown_ground_truth_clips_excluded": int(np.sum(~valid)),
    }


def decode_coarse_phone_events(
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    config: ActionDecoderConfig = ActionDecoderConfig(),
) -> list[dict[str, Any]]:
    if len(timestamps) == 0:
        return []
    if len(timestamps) != len(probabilities) or probabilities.shape[1] != len(COARSE_CLASSES):
        raise ValueError("expected aligned timestamps and three-class probabilities")
    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.5
    active_start: int | None = None
    scores: list[float] = []
    positive_samples = 0
    negative_run = 0
    events: list[dict[str, Any]] = []

    def close(end_index: int) -> None:
        nonlocal active_start, scores, positive_samples, negative_run
        if active_start is not None and positive_samples >= config.minimum_positive_samples:
            start = float(timestamps[active_start])
            end = float(timestamps[min(end_index, len(timestamps) - 1)])
            if end <= start:
                end = start + step
            events.append(
                {
                    "event_type": "PHONE",
                    "start_seconds": round(start, 4),
                    "end_seconds": round(end, 4),
                    "duration": round(end - start, 4),
                    "confidence": round(max(scores), 4),
                    "source": "V31_COARSE_CAUSAL_ACTION",
                }
            )
        active_start = None
        scores = []
        positive_samples = 0
        negative_run = 0

    for index, row in enumerate(probabilities):
        phone_score = float(row[PHONE_CLASS])
        candidate = int(np.argmax(row)) == PHONE_CLASS and phone_score >= config.threshold
        if active_start is None:
            if candidate:
                active_start = index
                scores = [phone_score]
                positive_samples = 1
            continue
        if candidate:
            scores.append(phone_score)
            positive_samples += 1
            negative_run = 0
        elif phone_score >= config.threshold * config.release_ratio:
            scores.append(phone_score)
            negative_run = 0
        else:
            negative_run += 1
            if negative_run >= config.release_patience:
                close(index)
    if active_start is not None:
        close(len(timestamps) - 1)

    merged: list[dict[str, Any]] = []
    for event in events:
        if merged and event["start_seconds"] <= merged[-1]["end_seconds"] + config.merge_gap_seconds:
            merged[-1]["end_seconds"] = event["end_seconds"]
            merged[-1]["duration"] = round(
                merged[-1]["end_seconds"] - merged[-1]["start_seconds"], 4
            )
            merged[-1]["confidence"] = max(merged[-1]["confidence"], event["confidence"])
        else:
            merged.append(dict(event))
    return merged


def _overlaps(left: dict[str, Any], right: dict[str, Any]) -> bool:
    return min(float(left["end_seconds"]), float(right["end_seconds"])) > max(
        float(left["start_seconds"]), float(right["start_seconds"])
    )


def gate_object_events(
    object_events: list[dict[str, Any]], action_events: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    return [dict(event) for event in object_events if any(_overlaps(event, action) for action in action_events)]


def recover_object_events(
    object_events: list[dict[str, Any]],
    action_events: list[dict[str, Any]],
    *,
    minimum_confidence: float,
    minimum_duration: float,
    merge_gap_seconds: float,
) -> list[dict[str, Any]]:
    combined = [dict(event) for event in object_events]
    for action in action_events:
        if (
            float(action["confidence"]) >= minimum_confidence
            and float(action["duration"]) >= minimum_duration
            and not any(_overlaps(action, obj) for obj in object_events)
        ):
            combined.append(dict(action))
    combined.sort(key=lambda row: (float(row["start_seconds"]), float(row["end_seconds"])))
    merged: list[dict[str, Any]] = []
    for event in combined:
        if merged and float(event["start_seconds"]) <= float(merged[-1]["end_seconds"]) + merge_gap_seconds:
            merged[-1]["end_seconds"] = max(
                float(merged[-1]["end_seconds"]), float(event["end_seconds"])
            )
            merged[-1]["duration"] = round(
                float(merged[-1]["end_seconds"]) - float(merged[-1]["start_seconds"]), 4
            )
            merged[-1]["confidence"] = max(
                float(merged[-1].get("confidence", 0.0)), float(event.get("confidence", 0.0))
            )
            merged[-1]["source"] = "D5_640_PLUS_V31_COARSE_RECOVERY"
        else:
            merged.append(dict(event))
    return merged


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _aggregate(per_subject: dict[str, dict[str, Any]], subjects: tuple[str, ...]) -> dict[str, Any]:
    tp = sum(per_subject[s]["true_positives"] for s in subjects)
    fp = sum(per_subject[s]["false_positives"] for s in subjects)
    fn = sum(per_subject[s]["false_negatives"] for s in subjects)
    minutes = sum(per_subject[s]["duration_seconds"] for s in subjects) / 60.0
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "subjects": list(subjects),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fp / max(minutes, 1e-9), 4),
        "evaluated_minutes": round(minutes, 4),
    }


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    require_operation("local_dmd_training")
    result_path = output_dir / "v31_coarse_results.json"
    if result_path.exists():
        raise FileExistsError("V31 coarse output is immutable; use a new output directory")
    if shutil.disk_usage(Path.cwd()).free < 100 * 1024 * 1024:
        raise OSError("at least 100 MiB free disk space is required")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    decoder_raw = config["decoder"]
    decoder = ActionDecoderConfig(
        threshold=float(decoder_raw["threshold"]),
        release_ratio=float(decoder_raw["release_ratio"]),
        release_patience=int(decoder_raw["release_patience"]),
        minimum_positive_samples=int(decoder_raw["minimum_positive_samples"]),
        merge_gap_seconds=float(decoder_raw["merge_gap_seconds"]),
    )
    subjects_order = tuple(DEVELOPMENT_SUBJECTS)
    benchmark_subjects = tuple(config["evaluation"]["benchmark_subjects"])
    root = discover_external_dmd_root()
    subjects_dir = root / "subjects"
    embedding_dir = Path(config["inputs"]["embedding_directory"])
    template = config["inputs"]["embedding_filename_template"]
    subjects: dict[str, SubjectEmbeddings] = {}
    truths: dict[str, list[dict[str, Any]]] = {}
    durations: dict[str, float] = {}
    inputs: list[dict[str, Any]] = []
    for subject_id in subjects_order:
        embedding_path = embedding_dir / template.format(subject=subject_id)
        feature_path = root / "derivatives" / f"v3_features_{subject_id}.json"
        subjects[subject_id] = coarse_subject(SubjectEmbeddings.load(subject_id, embedding_path))
        truths[subject_id] = _ground_truth(subjects_dir, subject_id)
        durations[subject_id] = float(json.loads(feature_path.read_text(encoding="utf-8"))["duration_seconds"])
        inputs.append(
            {
                "subject": subject_id,
                "embedding_path": str(embedding_path),
                "embedding_sha256": sha256_file(embedding_path),
                "records": len(subjects[subject_id].embeddings),
            }
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(exist_ok=True)
    checkpoint_dir = Path("models/candidate") / model_cfg["checkpoint_name"]
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    predictions: dict[str, dict[str, Any]] = {}
    action_events: dict[str, list[dict[str, Any]]] = {}
    action_metrics: dict[str, dict[str, Any]] = {}
    folds: list[dict[str, Any]] = []
    for fold_index, heldout in enumerate(subjects_order):
        training_subjects = tuple(s for s in subjects_order if s != heldout)
        train_dataset = CausalEmbeddingDataset(
            [subjects[s] for s in training_subjects],
            sequence_length=int(model_cfg["sequence_length"]),
            stride=int(model_cfg["training_stride"]),
            labeled_only=True,
            num_classes=len(COARSE_CLASSES),
        )
        model, training = train_causal_action_model(
            train_dataset,
            epochs=int(model_cfg["epochs"]),
            batch_size=int(model_cfg["batch_size"]),
            learning_rate=float(model_cfg["learning_rate"]),
            hidden_size=int(model_cfg["hidden_size"]),
            seed=int(model_cfg["seed"]) + fold_index,
            weighted_loss=bool(model_cfg["weighted_loss"]),
            num_classes=len(COARSE_CLASSES),
        )
        prediction = predict_causal_action(
            model,
            subjects[heldout],
            sequence_length=int(model_cfg["sequence_length"]),
            batch_size=int(model_cfg["batch_size"]),
        )
        events = decode_coarse_phone_events(prediction["timestamps"], prediction["probabilities"], decoder)
        metrics = event_metrics(truths[heldout], events, durations[heldout])
        metrics["duration_seconds"] = durations[heldout]
        checkpoint_path = checkpoint_dir / f"loso_{heldout}.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "architecture": model_cfg["architecture"],
                "class_names": list(COARSE_CLASSES),
                "class_mapping": {"SAFE_DRIVING": [0], "PHONE_USE": [1, 2, 3, 4], "OTHER_ACTIVITY": [5]},
                "heldout_subject": heldout,
                "training_subjects": training_subjects,
                "causal_contract": "PAST_AND_CURRENT_ONLY; TARGET_IS_FINAL_WINDOW_SAMPLE",
                "config": config,
                "git_head": _git_head(),
            },
            checkpoint_path,
        )
        prediction_path = folds_dir / f"loso_{heldout}_predictions.npz"
        np.savez_compressed(
            prediction_path,
            timestamps=prediction["timestamps"],
            labels=prediction["labels"],
            probabilities=prediction["probabilities"],
            targets=prediction["targets"],
        )
        predictions[heldout] = prediction
        action_events[heldout] = events
        action_metrics[heldout] = metrics
        folds.append(
            {
                "heldout_subject": heldout,
                "training_subjects": list(training_subjects),
                "training": training,
                "clip_metrics": coarse_classification_metrics(prediction["labels"], prediction["probabilities"]),
                "action_event_metrics": metrics,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "prediction_path": str(prediction_path),
                "prediction_sha256": sha256_file(prediction_path),
            }
        )

    pooled_clip = coarse_classification_metrics(
        np.concatenate([predictions[s]["labels"] for s in subjects_order]),
        np.concatenate([predictions[s]["probabilities"] for s in subjects_order]),
    )
    baseline_root = Path("reports/v28_local_raw_d5/imgsz640/subjects")
    gated_events: dict[str, list[dict[str, Any]]] = {}
    recovered_events: dict[str, list[dict[str, Any]]] = {}
    gated_metrics: dict[str, dict[str, Any]] = {}
    recovered_metrics: dict[str, dict[str, Any]] = {}
    for subject_id in benchmark_subjects:
        objects = _read_jsonl(baseline_root / subject_id / "event_predictions.jsonl")
        gated_events[subject_id] = gate_object_events(objects, action_events[subject_id])
        recovered_events[subject_id] = recover_object_events(
            objects,
            action_events[subject_id],
            minimum_confidence=float(decoder_raw["recovery_minimum_confidence"]),
            minimum_duration=float(decoder_raw["recovery_minimum_duration_seconds"]),
            merge_gap_seconds=float(decoder_raw["merge_gap_seconds"]),
        )
        gated_metrics[subject_id] = event_metrics(
            truths[subject_id], gated_events[subject_id], durations[subject_id]
        )
        recovered_metrics[subject_id] = event_metrics(
            truths[subject_id], recovered_events[subject_id], durations[subject_id]
        )
        gated_metrics[subject_id]["duration_seconds"] = durations[subject_id]
        recovered_metrics[subject_id]["duration_seconds"] = durations[subject_id]

    result = {
        "status": "COMPLETED_DEVELOPMENT_ONLY",
        "created_at": datetime.now(UTC).isoformat(),
        "git_head": _git_head(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "registration_path": "reports/v31_ai85_plan_execution/coarse_action_experiment_registration.json",
        "class_names": list(COARSE_CLASSES),
        "input_provenance": inputs,
        "input_set_sha256": stable_json_hash(inputs),
        "causal_contract": {
            "window": "PAST_AND_CURRENT_ONLY",
            "target_position": "FINAL_SAMPLE",
            "recurrent_direction": "UNIDIRECTIONAL",
            "future_frames": 0,
        },
        "folds": folds,
        "pooled_clip_metrics_all_development": pooled_clip,
        "action_only": {
            "all_development": _aggregate(action_metrics, subjects_order),
            "benchmark_gc_gz": _aggregate(action_metrics, benchmark_subjects),
            "per_subject": action_metrics,
            "events": action_events,
        },
        "object_gated_by_coarse_action": {
            "benchmark_gc_gz": _aggregate(gated_metrics, benchmark_subjects),
            "per_subject": gated_metrics,
            "events": gated_events,
        },
        "object_plus_coarse_recovery": {
            "benchmark_gc_gz": _aggregate(recovered_metrics, benchmark_subjects),
            "per_subject": recovered_metrics,
            "events": recovered_events,
        },
        "promotion_gate": {
            "required_precision_recall_f1": float(config["evaluation"]["promotion_threshold"]),
            "maximum_false_positives": int(config["evaluation"]["maximum_false_positives"]),
            "eligible": False,
            "reason": "Development-only outer LOSO; thresholds frozen and independent holdout still required.",
        },
        "governance": {
            "dmd_training_location": "LOCAL_ONLY",
            "dmd_artifacts_uploaded_to_kaggle": False,
            "consumed_holdouts_accessed": [],
        },
    }
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = run(args.config, args.output)
    print(
        json.dumps(
            {
                "status": result["status"],
                "pooled_clip": result["pooled_clip_metrics_all_development"],
                "action_only": result["action_only"]["benchmark_gc_gz"],
                "gated": result["object_gated_by_coarse_action"]["benchmark_gc_gz"],
                "recovery": result["object_plus_coarse_recovery"]["benchmark_gc_gz"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
