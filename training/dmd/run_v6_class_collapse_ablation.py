"""Run controlled V6 class-collapse ablations on the frozen V5 embeddings."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from training.common import sha256_file, stable_json_hash
from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS
from training.dmd.run_v5_experiments import (
    CONSUMED_HOLDOUTS,
    _evaluate_predictions,
    _ground_truth,
    discover_external_dmd_root,
)
from training.dmd.v5_resnet_gru import (
    EmbeddingClipDataset,
    SubjectEmbeddings,
    predict_embeddings,
    train_resnet_gru,
)
from training.dmd.v5_video_action import (
    ACTION_CLASSES,
    classification_metrics,
    merge_evidence_events,
    probabilities_to_events,
)
from training.dmd.v6_action_hierarchy import (
    COARSE_CLASSES,
    predict_hierarchical,
    train_hierarchical_resnet_gru,
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _find_experiment(result: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    return next(row for row in result["experiments"] if row["experiment_id"] == experiment_id)


def _coarse_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    valid = labels >= 0
    truth_flat = labels[valid]
    truth = np.where(truth_flat == 0, 0, np.where(truth_flat == 5, 2, 1))
    probs = probabilities[valid]
    coarse_probs = np.column_stack((probs[:, 0], probs[:, 1:5].sum(axis=1), probs[:, 5]))
    predicted = coarse_probs.argmax(axis=1)
    confusion = np.zeros((len(COARSE_CLASSES), len(COARSE_CLASSES)), dtype=np.int64)
    for expected, actual in zip(truth, predicted, strict=True):
        confusion[int(expected), int(actual)] += 1
    per_class: dict[str, dict[str, Any]] = {}
    f1_values: list[float] = []
    recalls: list[float] = []
    for index, name in enumerate(COARSE_CLASSES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[name] = {
            "support": int(confusion[index, :].sum()),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
        f1_values.append(f1)
        recalls.append(recall)
    return {
        "accuracy": round(float(np.mean(predicted == truth)), 4),
        "macro_f1": round(float(np.mean(f1_values)), 4),
        "balanced_accuracy": round(float(np.mean(recalls)), 4),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_order": list(COARSE_CLASSES),
        "evaluated_clips": int(len(truth)),
    }


def _class_and_shift_audit(
    embeddings: dict[str, SubjectEmbeddings], derivatives: Path, sequence_length: int
) -> dict[str, Any]:
    per_subject: dict[str, Any] = {}
    class_centroids: dict[str, dict[int, np.ndarray]] = {}
    for subject_id, subject in embeddings.items():
        labeled = subject.labels[subject.labels >= 0]
        frame_counts = np.bincount(labeled, minlength=len(ACTION_CLASSES))
        dataset = EmbeddingClipDataset(
            [subject], sequence_length=sequence_length, stride=2, labeled_only=True
        )
        clip_labels = np.asarray(
            [int(dataset.subjects[s].labels[center]) for s, _, center in dataset.items]
        )
        clip_counts = np.bincount(clip_labels, minlength=len(ACTION_CLASSES))
        normalized = subject.embeddings / np.maximum(
            np.linalg.norm(subject.embeddings, axis=1, keepdims=True), 1e-9
        )
        adjacent_cosine = np.sum(normalized[:-1] * normalized[1:], axis=1)
        unique_records = len(np.unique(subject.embeddings, axis=0))
        rgb_path = derivatives / "v5_rgb_cache" / f"v5_rgb_{subject_id}_64.npz"
        with np.load(rgb_path, allow_pickle=False) as payload:
            frames = payload["frames"].astype(np.float32) / 255.0
        luminance = (
            0.2126 * frames[..., 0] + 0.7152 * frames[..., 1] + 0.0722 * frames[..., 2]
        )
        temporal_abs_difference = np.abs(frames[1:] - frames[:-1]).mean(axis=(1, 2, 3))
        per_subject[subject_id] = {
            "records": len(subject.labels),
            "labeled_records": int(len(labeled)),
            "frame_class_counts": dict(zip(ACTION_CLASSES, frame_counts.tolist(), strict=True)),
            "training_clip_class_counts": dict(
                zip(ACTION_CLASSES, clip_counts.tolist(), strict=True)
            ),
            "exact_duplicate_embeddings": int(len(subject.embeddings) - unique_records),
            "adjacent_embedding_cosine_quantiles": {
                str(q): round(float(np.quantile(adjacent_cosine, q)), 4)
                for q in (0.1, 0.5, 0.9)
            },
            "roi_luminance_mean": round(float(luminance.mean()), 4),
            "roi_luminance_std": round(float(luminance.std()), 4),
            "adjacent_rgb_absolute_difference_quantiles": {
                str(q): round(float(np.quantile(temporal_abs_difference, q)), 4)
                for q in (0.1, 0.5, 0.9)
            },
        }
        class_centroids[subject_id] = {}
        for class_index in range(len(ACTION_CLASSES)):
            values = normalized[subject.labels == class_index]
            if len(values):
                centroid = values.mean(axis=0)
                class_centroids[subject_id][class_index] = centroid / max(
                    float(np.linalg.norm(centroid)), 1e-9
                )
    centroid_distances: list[dict[str, Any]] = []
    for class_index, class_name in enumerate(ACTION_CLASSES):
        for left_index, left in enumerate(DEVELOPMENT_SUBJECTS):
            for right in DEVELOPMENT_SUBJECTS[left_index + 1 :]:
                if (
                    class_index not in class_centroids[left]
                    or class_index not in class_centroids[right]
                ):
                    continue
                cosine = float(
                    np.dot(class_centroids[left][class_index], class_centroids[right][class_index])
                )
                centroid_distances.append(
                    {
                        "class": class_name,
                        "subjects": [left, right],
                        "cosine_distance": round(1.0 - cosine, 4),
                    }
                )
    return {
        "per_subject": per_subject,
        "training_window_overlap": {
            "sequence_length": sequence_length,
            "stride": 2,
            "shared_frames_between_adjacent_training_clips": sequence_length - 2,
            "overlap_fraction": round((sequence_length - 2) / sequence_length, 4),
            "interpretation": (
                "Expected temporal overlap within a subject; outer LOSO prevents "
                "cross-subject leakage."
            ),
        },
        "same_class_subject_centroid_cosine_distances": centroid_distances,
    }


def _false_positive_audit(
    experiment: dict[str, Any], predictions: dict[str, dict[str, Any]], subjects_dir: Path
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for subject_id in DEVELOPMENT_SUBJECTS:
        timestamps = predictions[subject_id]["timestamps"]
        labels = predictions[subject_id]["labels"]
        ground_truth = _ground_truth(subjects_dir, subject_id)
        false_positives = experiment["per_subject"][subject_id]["unmatched_predictions"]
        for index, event in enumerate(false_positives, start=1):
            start = float(event["start_seconds"])
            end = float(event["end_seconds"])
            mask = (timestamps >= start) & (timestamps <= end) & (labels >= 0)
            interval_labels = labels[mask]
            denominator = max(len(interval_labels), 1)
            safe_fraction = float(np.sum(interval_labels == 0) / denominator)
            phone_fraction = float(
                np.sum((interval_labels >= 1) & (interval_labels <= 4)) / denominator
            )
            other_fraction = float(np.sum(interval_labels == 5) / denominator)
            overlapping_truth = [
                item
                for item in ground_truth
                if min(end, float(item["end_seconds"]))
                > max(start, float(item["start_seconds"]))
            ]
            sources = event.get("sources", [event.get("source", "UNKNOWN")])
            if overlapping_truth:
                category = "TEMPORAL_OVERMERGE_OR_MATCHING_MISMATCH"
            elif sources == ["OBJECT_GEOMETRY"]:
                category = "OBJECT_GEOMETRY_FALSE_ACTIVATION"
            elif safe_fraction >= 0.5:
                category = "SAFE_DRIVING_MISCLASSIFIED"
            elif other_fraction >= 0.5:
                category = "OTHER_ACTIVITY_MISCLASSIFIED"
            elif phone_fraction >= 0.5:
                category = "PHONE_ACTION_OUTSIDE_EVALUATED_EVENT"
            else:
                category = "INSUFFICIENT_LABEL_EVIDENCE"
            rows.append(
                {
                    "case_id": f"{experiment['experiment_id']}_{subject_id}_FP_{index:02d}",
                    "subject_id": subject_id,
                    "prediction": event,
                    "sampled_label_fractions": {
                        "safe_driving": round(safe_fraction, 4),
                        "phone_use": round(phone_fraction, 4),
                        "other_activity": round(other_fraction, 4),
                        "samples": int(len(interval_labels)),
                    },
                    "overlapping_ground_truth_events": overlapping_truth,
                    "primary_category": category,
                    "visual_cause_status": "NOT_ESTABLISHED_FROM_EMBEDDING_CACHE",
                }
            )
    return rows


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    backbone = config["backbone"]
    event_cfg = config["event_decision"]
    data_root = discover_external_dmd_root()
    derivatives = data_root / "derivatives"
    subjects_dir = data_root / "subjects"
    embedding_dir = derivatives / "v5_resnet18_embeddings"
    checkpoint_dir = Path("models/candidate/v6_action_ablation_001")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)
    v5 = json.loads(
        Path("reports/v5_research_upgrade/v5_experiment_results_round2.json").read_text(
            encoding="utf-8"
        )
    )
    embeddings = {
        subject_id: SubjectEmbeddings.load(
            subject_id, embedding_dir / f"v5_resnet18_{subject_id}.npz"
        )
        for subject_id in DEVELOPMENT_SUBJECTS
    }
    durations = {
        subject_id: float(
            json.loads(
                (derivatives / f"v3_features_{subject_id}.json").read_text(encoding="utf-8")
            )["duration_seconds"]
        )
        for subject_id in DEVELOPMENT_SUBJECTS
    }
    object_experiment = _find_experiment(v5, "B0_V4_CURRENT")
    object_events = {
        subject_id: object_experiment["per_subject"][subject_id]["evaluation_evidence"][
            "predicted_events"
        ]
        for subject_id in DEVELOPMENT_SUBJECTS
    }

    variants: list[dict[str, Any]] = []
    prediction_sets: dict[str, dict[str, dict[str, Any]]] = {}
    for variant_index, variant in enumerate(config["ablation_variants"]):
        heldout_predictions: dict[str, dict[str, Any]] = {}
        fold_rows: list[dict[str, Any]] = []
        for fold_index, heldout_subject in enumerate(DEVELOPMENT_SUBJECTS):
            fold_stem = f"{variant['id'].lower()}_loso_{heldout_subject}"
            fold_result_path = output_dir / "folds" / f"{fold_stem}.json"
            fold_prediction_path = output_dir / "folds" / f"{fold_stem}.npz"
            if fold_result_path.is_file() and fold_prediction_path.is_file():
                fold_result = json.loads(fold_result_path.read_text(encoding="utf-8"))
                with np.load(fold_prediction_path, allow_pickle=False) as payload:
                    heldout_predictions[heldout_subject] = {
                        "centers": payload["centers"],
                        "timestamps": payload["timestamps"],
                        "labels": payload["labels"],
                        "probabilities": payload["probabilities"],
                        "events": fold_result["events"],
                    }
                fold_rows.append(
                    {
                        key: value
                        for key, value in fold_result.items()
                        if key not in {"events", "prediction_path", "prediction_sha256"}
                    }
                )
                continue
            training_subjects = tuple(
                subject_id
                for subject_id in DEVELOPMENT_SUBJECTS
                if subject_id != heldout_subject
            )
            train_dataset = EmbeddingClipDataset(
                [embeddings[item] for item in training_subjects],
                sequence_length=int(backbone["sequence_length"]),
                stride=int(backbone["training_stride"]),
                labeled_only=True,
            )
            checkpoint_path = checkpoint_dir / f"{variant['id'].lower()}_loso_{heldout_subject}.pt"
            seed = int(backbone["seed"]) + variant_index * 100 + fold_index
            if variant["architecture"] == "hierarchical":
                model, training = train_hierarchical_resnet_gru(
                    train_dataset,
                    epochs=int(backbone["epochs"]),
                    batch_size=int(backbone["batch_size"]),
                    learning_rate=float(backbone["learning_rate"]),
                    seed=seed,
                    loss_mode=variant["loss_mode"],
                )
                heldout_dataset = EmbeddingClipDataset(
                    [embeddings[heldout_subject]],
                    sequence_length=int(backbone["sequence_length"]),
                    stride=1,
                    labeled_only=False,
                )
                heldout = predict_hierarchical(
                    model, heldout_dataset, batch_size=int(backbone["batch_size"])
                )
            else:
                model, training = train_resnet_gru(
                    train_dataset,
                    epochs=int(backbone["epochs"]),
                    batch_size=int(backbone["batch_size"]),
                    learning_rate=float(backbone["learning_rate"]),
                    seed=seed,
                    sampling_mode=variant["sampling_mode"],
                    loss_mode=variant["loss_mode"],
                )
                heldout = predict_embeddings(
                    model,
                    embeddings[heldout_subject],
                    sequence_length=int(backbone["sequence_length"]),
                    batch_size=int(backbone["batch_size"]),
                )
            threshold = float(event_cfg["fixed_action_threshold"])
            heldout["events"] = probabilities_to_events(
                heldout["timestamps"],
                heldout["probabilities"],
                threshold=threshold,
                release_threshold=threshold * float(event_cfg["release_ratio"]),
                minimum_samples=int(event_cfg["minimum_positive_samples"]),
                merge_gap_seconds=float(event_cfg["merge_gap_seconds"]),
            )
            heldout_predictions[heldout_subject] = heldout
            checkpoint_payload = {
                "model_state_dict": model.state_dict(),
                "variant": variant,
                "heldout_subject": heldout_subject,
                "training_subjects": training_subjects,
                "action_threshold": threshold,
                "threshold_provenance": event_cfg["threshold_provenance"],
                "dataset_role": config["dataset_role"],
                "forbidden_consumed_holdouts": CONSUMED_HOLDOUTS,
                "git_head": _git_head(),
            }
            torch.save(checkpoint_payload, checkpoint_path)
            fold_rows.append(
                {
                    "heldout_subject": heldout_subject,
                    "training_subjects": list(training_subjects),
                    "training": training,
                    "checkpoint_path": str(checkpoint_path),
                    "checkpoint_sha256": sha256_file(checkpoint_path),
                    "fixed_action_threshold": threshold,
                }
            )
        labels = np.concatenate(
            [heldout_predictions[item]["labels"] for item in DEVELOPMENT_SUBJECTS]
        )
        probabilities = np.concatenate(
            [heldout_predictions[item]["probabilities"] for item in DEVELOPMENT_SUBJECTS], axis=0
        )
        clip_metrics = classification_metrics(labels, probabilities)
        coarse_metrics = _coarse_metrics(labels, probabilities)
        action_events = {
            subject_id: heldout_predictions[subject_id]["events"]
            for subject_id in DEVELOPMENT_SUBJECTS
        }
        action_pooled, action_per_subject = _evaluate_predictions(
            subjects_dir, durations, action_events
        )
        fused_events = {
            subject_id: merge_evidence_events(
                object_events[subject_id], action_events[subject_id]
            )
            for subject_id in DEVELOPMENT_SUBJECTS
        }
        fused_pooled, fused_per_subject = _evaluate_predictions(
            subjects_dir, durations, fused_events
        )
        variants.append(
            {
                "variant": variant,
                "clip_metrics": clip_metrics,
                "coarse_metrics": coarse_metrics,
                "folds": fold_rows,
                "experiments": [
                    {
                        "experiment_id": f"V6_{variant['id']}_ACTION",
                        "kind": "ACTION_ONLY",
                        "pooled_metrics": action_pooled,
                        "per_subject": action_per_subject,
                    },
                    {
                        "experiment_id": f"V6_{variant['id']}_FUSION",
                        "kind": "V4_OBJECT_PLUS_ACTION",
                        "pooled_metrics": fused_pooled,
                        "per_subject": fused_per_subject,
                    },
                ],
            }
        )
        prediction_sets[variant["id"]] = heldout_predictions

    selection_rows: list[dict[str, Any]] = []
    for variant_result in variants:
        safe_recall = variant_result["coarse_metrics"]["per_class"]["SAFE_DRIVING"]["recall"]
        for experiment in variant_result["experiments"]:
            worst_subject_fa = max(
                row["false_alarms_per_minute"]
                for row in experiment["per_subject"].values()
            )
            metrics = experiment["pooled_metrics"]
            gates = {
                "aggregate_fa": metrics["false_alarms_per_minute"]
                <= float(config["selection"]["maximum_false_alarms_per_minute"]),
                "worst_subject_fa": worst_subject_fa
                <= float(
                    config["selection"]["maximum_worst_subject_false_alarms_per_minute"]
                ),
                "safe_driving_recall": safe_recall
                >= float(config["selection"]["minimum_safe_driving_recall"]),
            }
            selection_rows.append(
                {
                    "experiment_id": experiment["experiment_id"],
                    "variant_id": variant_result["variant"]["id"],
                    "kind": experiment["kind"],
                    "pooled_metrics": metrics,
                    "safe_driving_recall": safe_recall,
                    "worst_subject_false_alarms_per_minute": round(worst_subject_fa, 4),
                    "gates": gates,
                    "eligible": all(gates.values()),
                }
            )
    eligible = [row for row in selection_rows if row["eligible"]]
    selected = (
        max(
            eligible,
            key=lambda row: (
                row["pooled_metrics"]["f1"],
                row["pooled_metrics"]["recall"],
                -row["pooled_metrics"]["false_alarms_per_minute"],
            ),
        )
        if eligible
        else None
    )
    false_positive_audits: dict[str, Any] = {}
    for variant_result in variants:
        variant_id = variant_result["variant"]["id"]
        for experiment in variant_result["experiments"]:
            if experiment["kind"] == "V4_OBJECT_PLUS_ACTION":
                false_positive_audits[experiment["experiment_id"]] = _false_positive_audit(
                    experiment, prediction_sets[variant_id], subjects_dir
                )

    metadata = {
        "schema_version": 1,
        "model_family": "ROADWATCH_V6_ACTION_ABLATION_001",
        "activation_state": "DEVELOPMENT_ONLY_NOT_VALIDATED",
        "human_review_required": True,
        "production_approved": False,
        "dataset_role": config["dataset_role"],
        "forbidden_consumed_holdouts": list(CONSUMED_HOLDOUTS),
        "checkpoint_count": sum(len(row["folds"]) for row in variants),
        "variants": [row["variant"] for row in variants],
    }
    metadata_path = checkpoint_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    result = {
        "status": "COMPLETED",
        "created_at": datetime.now(UTC).isoformat(),
        "git_head": _git_head(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "dataset_role": config["dataset_role"],
        "development_subjects": list(DEVELOPMENT_SUBJECTS),
        "forbidden_consumed_holdouts": list(CONSUMED_HOLDOUTS),
        "holdouts_accessed": [],
        "evaluation_protocol": (
            "OUTER_LOSO; fixed threshold 0.80 pre-registered from V5; no per-variant "
            "outer-label threshold tuning; iterative development on the same three subjects"
        ),
        "calibration_audit": {
            "v5_status": "IN_SAMPLE_TRAINING_SUBJECT_CALIBRATION",
            "risk": (
                "Threshold predictions came from the same subjects used to fit each outer-fold "
                "model, so they are training-only but optimistically biased."
            ),
            "v6_control": event_cfg["threshold_provenance"],
            "nested_validation_limit": (
                "Only two training subjects exist inside each outer fold; a two-way inner split "
                "would be high variance and is not claimed as reliable nested validation."
            ),
        },
        "data_audit": _class_and_shift_audit(
            embeddings, derivatives, int(backbone["sequence_length"])
        ),
        "v5_comparators": {
            experiment_id: _find_experiment(v5, experiment_id)["pooled_metrics"]
            for experiment_id in (
                "B2_TINY_CNN_TEMPORAL",
                "B3_RESNET18_GRU",
                "B8_B3_LATE_FUSION",
            )
        },
        "variants": variants,
        "selection_rule": config["selection"],
        "selection_rows": selection_rows,
        "selected_candidate": selected,
        "false_positive_audits": false_positive_audits,
        "checkpoint_metadata_path": str(metadata_path),
        "checkpoint_metadata_sha256": sha256_file(metadata_path),
        "result_input_hash": stable_json_hash(
            {
                subject_id: sha256_file(
                    embedding_dir / f"v5_resnet18_{subject_id}.npz"
                )
                for subject_id in DEVELOPMENT_SUBJECTS
            }
        ),
        "governance": {
            "activation_state": "DEVELOPMENT_ONLY_NOT_VALIDATED",
            "action_event_status": "NEEDS_REVIEW",
            "fusion_event_status": "PENDING",
            "automatic_violation": False,
            "production_changed": False,
        },
    }
    output_path = output_dir / "v6_class_collapse_ablation.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Roadwatch V6 class-collapse ablations")
    parser.add_argument(
        "--config", type=Path, default=Path("models/model_config_v6_completion.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/v6_completion")
    )
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    selected = result["selected_candidate"]
    print(json.dumps({"status": result["status"], "selected_candidate": selected}, indent=2))


if __name__ == "__main__":
    main()
