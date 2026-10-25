"""Train and evaluate the V30 local-only causal DMD action branch."""

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
    CausalActionGRU,
    CausalEmbeddingDataset,
    CLASS_TO_SOURCE_ACTION,
    class_aware_event_metrics,
    decode_action_events,
    predict_causal_action,
    train_causal_action_model,
)
from training.dmd.v5_resnet_gru import SubjectEmbeddings
from training.dmd.v5_video_action import classification_metrics


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


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


def _merge_same_action(events: list[dict[str, Any]], gap_seconds: float = 0.15) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    for event in sorted(events, key=lambda row: (row["start_seconds"], row["end_seconds"])):
        if (
            merged
            and merged[-1]["source_action"] == event["source_action"]
            and event["start_seconds"] <= merged[-1]["end_seconds"] + gap_seconds
        ):
            merged[-1]["end_seconds"] = max(
                merged[-1]["end_seconds"], event["end_seconds"]
            )
            merged[-1]["duration"] = round(
                merged[-1]["end_seconds"] - merged[-1]["start_seconds"], 4
            )
            merged[-1]["confidence"] = max(
                merged[-1]["confidence"], event["confidence"]
            )
        else:
            merged.append(dict(event))
    return merged


def label_object_events_with_actions(
    object_events: list[dict[str, Any]],
    action_events: list[dict[str, Any]],
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    *,
    add_action_only_recovery: bool,
) -> list[dict[str, Any]]:
    """Split detector events with causal actions, retaining object evidence as the anchor."""

    labeled: list[dict[str, Any]] = []
    for object_event in object_events:
        start = float(object_event["start_seconds"])
        end = float(object_event["end_seconds"])
        overlaps = [
            row
            for row in action_events
            if min(end, float(row["end_seconds"]))
            > max(start, float(row["start_seconds"]))
        ]
        if overlaps:
            for row in overlaps:
                local_start = max(start, float(row["start_seconds"]))
                local_end = min(end, float(row["end_seconds"]))
                labeled.append(
                    {
                        "event_type": "PHONE",
                        "source_action": row["source_action"],
                        "start_seconds": round(local_start, 4),
                        "end_seconds": round(local_end, 4),
                        "duration": round(local_end - local_start, 4),
                        "confidence": round(
                            max(float(object_event.get("confidence", 0.0)), row["confidence"]),
                            4,
                        ),
                        "source": "D5_640_OBJECT_PLUS_V30_SEGMENT",
                    }
                )
            continue
        mask = (timestamps >= start) & (timestamps <= end)
        if not np.any(mask):
            nearest = int(np.argmin(np.abs(timestamps - (start + end) / 2.0)))
            mask = np.zeros(len(timestamps), dtype=bool)
            mask[nearest] = True
        mean_phone = probabilities[mask][:, 1:5].mean(axis=0)
        class_id = int(np.argmax(mean_phone)) + 1
        labeled.append(
            {
                "event_type": "PHONE",
                "source_action": CLASS_TO_SOURCE_ACTION[class_id],
                "start_seconds": start,
                "end_seconds": end,
                "duration": round(end - start, 4),
                "confidence": round(float(mean_phone[class_id - 1]), 4),
                "source": "D5_640_OBJECT_PLUS_V30_DOMINANT_LABEL",
            }
        )
    if add_action_only_recovery:
        for row in action_events:
            overlaps_object = any(
                min(float(row["end_seconds"]), float(obj["end_seconds"]))
                > max(float(row["start_seconds"]), float(obj["start_seconds"]))
                for obj in object_events
            )
            if not overlaps_object and row["confidence"] >= 0.70 and row["duration"] >= 1.0:
                recovered = dict(row)
                recovered["source"] = "V30_HIGH_CONFIDENCE_ACTION_RECOVERY"
                labeled.append(recovered)
    return _merge_same_action(labeled)


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    require_operation("local_dmd_training")
    if (output_dir / "v30_results.json").exists():
        raise FileExistsError("V30 output is immutable; use a new output directory")
    if shutil.disk_usage(Path.cwd()).free < 100 * 1024 * 1024:
        raise OSError("at least 100 MiB free disk space is required")
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    model_cfg = config["model"]
    input_cfg = config.get("inputs", {})
    decoder_cfg = ActionDecoderConfig(
        **{
            key: config["decoder"][key]
            for key in (
                "threshold",
                "release_ratio",
                "release_patience",
                "minimum_positive_samples",
                "merge_gap_seconds",
            )
        }
    )
    subjects_order = tuple(DEVELOPMENT_SUBJECTS)
    if set(subjects_order) != {"gB-9", "gC-14", "gZ-36"}:
        raise ValueError("unexpected development subject allowlist")
    benchmark_subjects = tuple(config["evaluation"]["benchmark_subjects"])
    root = discover_external_dmd_root()
    derivatives = root / "derivatives"
    subjects_dir = root / "subjects"
    output_dir.mkdir(parents=True, exist_ok=True)
    folds_dir = output_dir / "folds"
    folds_dir.mkdir(exist_ok=True)
    checkpoint_name = model_cfg.get(
        "checkpoint_name",
        (
            "v30_causal_action_weighted_001"
            if bool(model_cfg["weighted_loss"])
            else "v30_causal_action_unweighted_001"
        ),
    )
    checkpoint_dir = Path("models/candidate") / checkpoint_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    subjects: dict[str, SubjectEmbeddings] = {}
    durations: dict[str, float] = {}
    inputs: list[dict[str, Any]] = []
    truths: dict[str, list[dict[str, Any]]] = {}
    embedding_directory = Path(
        input_cfg.get(
            "embedding_directory",
            derivatives / "v5_resnet18_embeddings",
        )
    )
    embedding_template = input_cfg.get(
        "embedding_filename_template", "v5_resnet18_{subject}.npz"
    )
    for subject in subjects_order:
        embedding_path = embedding_directory / embedding_template.format(subject=subject)
        feature_path = derivatives / f"v3_features_{subject}.json"
        if not embedding_path.is_file() or not feature_path.is_file():
            raise FileNotFoundError(f"missing local cache for {subject}")
        subjects[subject] = SubjectEmbeddings.load(subject, embedding_path)
        durations[subject] = float(
            json.loads(feature_path.read_text(encoding="utf-8"))["duration_seconds"]
        )
        truths[subject] = _ground_truth(subjects_dir, subject)
        inputs.append(
            {
                "subject": subject,
                "embedding_path": str(embedding_path),
                "embedding_sha256": sha256_file(embedding_path),
                "records": len(subjects[subject].embeddings),
            }
        )

    heldout_predictions: dict[str, dict[str, Any]] = {}
    heldout_events: dict[str, list[dict[str, Any]]] = {}
    folds: list[dict[str, Any]] = []
    for fold_index, heldout in enumerate(subjects_order):
        training_subjects = tuple(subject for subject in subjects_order if subject != heldout)
        train_dataset = CausalEmbeddingDataset(
            [subjects[subject] for subject in training_subjects],
            sequence_length=int(model_cfg["sequence_length"]),
            stride=int(model_cfg["training_stride"]),
            labeled_only=True,
        )
        model, training = train_causal_action_model(
            train_dataset,
            epochs=int(model_cfg["epochs"]),
            batch_size=int(model_cfg["batch_size"]),
            learning_rate=float(model_cfg["learning_rate"]),
            hidden_size=int(model_cfg["hidden_size"]),
            seed=int(model_cfg["seed"]) + fold_index,
            weighted_loss=bool(model_cfg["weighted_loss"]),
        )
        prediction = predict_causal_action(
            model,
            subjects[heldout],
            sequence_length=int(model_cfg["sequence_length"]),
            batch_size=int(model_cfg["batch_size"]),
        )
        events = decode_action_events(
            prediction["timestamps"], prediction["probabilities"], decoder_cfg
        )
        checkpoint_path = checkpoint_dir / f"loso_{heldout}.pt"
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "architecture": model_cfg["architecture"],
            "class_names": [
                "SAFE_DRIVING",
                "TEXTING_LEFT",
                "TEXTING_RIGHT",
                "PHONECALL_LEFT",
                "PHONECALL_RIGHT",
                "OTHER_ACTIVITY",
            ],
            "causal_contract": "PAST_AND_CURRENT_ONLY; TARGET_IS_FINAL_WINDOW_SAMPLE",
            "heldout_subject": heldout,
            "training_subjects": training_subjects,
            "config": config,
            "input_hashes": {
                row["subject"]: row["embedding_sha256"]
                for row in inputs
                if row["subject"] in training_subjects
            },
            "git_head": _git_head(),
        }
        torch.save(checkpoint_payload, checkpoint_path)
        np.savez_compressed(
            folds_dir / f"loso_{heldout}_predictions.npz",
            timestamps=prediction["timestamps"],
            labels=prediction["labels"],
            probabilities=prediction["probabilities"],
            targets=prediction["targets"],
        )
        fold_metrics = class_aware_event_metrics(
            truths[heldout], events, duration_seconds=durations[heldout]
        )
        fold_metrics["duration_seconds"] = durations[heldout]
        heldout_predictions[heldout] = prediction
        heldout_events[heldout] = events
        folds.append(
            {
                "heldout_subject": heldout,
                "training_subjects": list(training_subjects),
                "training": training,
                "clip_metrics": classification_metrics(
                    prediction["labels"], prediction["probabilities"]
                ),
                "action_event_metrics": fold_metrics,
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
                "prediction_path": str(folds_dir / f"loso_{heldout}_predictions.npz"),
                "prediction_sha256": sha256_file(
                    folds_dir / f"loso_{heldout}_predictions.npz"
                ),
            }
        )

    action_per_subject = {
        fold["heldout_subject"]: fold["action_event_metrics"] for fold in folds
    }
    pooled_clip = classification_metrics(
        np.concatenate([heldout_predictions[s]["labels"] for s in subjects_order]),
        np.concatenate([heldout_predictions[s]["probabilities"] for s in subjects_order]),
    )
    fusion_per_subject: dict[str, dict[str, Any]] = {}
    recovery_per_subject: dict[str, dict[str, Any]] = {}
    fusion_events: dict[str, list[dict[str, Any]]] = {}
    recovery_events: dict[str, list[dict[str, Any]]] = {}
    baseline_root = Path("reports/v28_local_raw_d5/imgsz640/subjects")
    for subject in benchmark_subjects:
        object_events = _read_jsonl(baseline_root / subject / "event_predictions.jsonl")
        prediction = heldout_predictions[subject]
        fusion_events[subject] = label_object_events_with_actions(
            object_events,
            heldout_events[subject],
            prediction["timestamps"],
            prediction["probabilities"],
            add_action_only_recovery=False,
        )
        recovery_events[subject] = label_object_events_with_actions(
            object_events,
            heldout_events[subject],
            prediction["timestamps"],
            prediction["probabilities"],
            add_action_only_recovery=True,
        )
        fusion_per_subject[subject] = class_aware_event_metrics(
            truths[subject], fusion_events[subject], duration_seconds=durations[subject]
        )
        recovery_per_subject[subject] = class_aware_event_metrics(
            truths[subject], recovery_events[subject], duration_seconds=durations[subject]
        )
        fusion_per_subject[subject]["duration_seconds"] = durations[subject]
        recovery_per_subject[subject]["duration_seconds"] = durations[subject]

    result = {
        "status": "COMPLETED_DEVELOPMENT_ONLY",
        "created_at": datetime.now(UTC).isoformat(),
        "git_head": _git_head(),
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path),
        "input_configuration": {
            "embedding_directory": str(embedding_directory),
            "embedding_filename_template": embedding_template,
            "preprocessing_id": input_cfg.get(
                "preprocessing_id", "LEGACY_64_TO_112"
            ),
        },
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
            "all_development": _aggregate(action_per_subject, subjects_order),
            "benchmark_gc_gz": _aggregate(action_per_subject, benchmark_subjects),
            "per_subject": action_per_subject,
        },
        "object_plus_action_segmentation": {
            "object_baseline": str(baseline_root),
            "benchmark_gc_gz": _aggregate(fusion_per_subject, benchmark_subjects),
            "per_subject": fusion_per_subject,
            "events": fusion_events,
        },
        "object_plus_high_confidence_action_recovery": {
            "object_baseline": str(baseline_root),
            "benchmark_gc_gz": _aggregate(recovery_per_subject, benchmark_subjects),
            "per_subject": recovery_per_subject,
            "events": recovery_events,
        },
        "promotion_gate": {
            "required_precision_recall_f1": float(
                config["evaluation"]["promotion_threshold"]
            ),
            "maximum_false_positives": int(
                config["evaluation"]["maximum_false_positives"]
            ),
            "independent_holdout_required": True,
            "eligible": False,
            "reason": "Development-only LOSO; independent holdout remains required.",
        },
        "governance": {
            "dmd_training_location": "LOCAL_ONLY",
            "dmd_artifacts_uploaded_to_kaggle": False,
            "consumed_holdouts_accessed": [],
        },
    }
    result_path = output_dir / "v30_results.json"
    result_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("models/model_config_v30_causal_action.yaml")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v30_causal_action")
    )
    args = parser.parse_args()
    result = run(args.config, args.output)
    summary = {
        "status": result["status"],
        "action_only": result["action_only"]["benchmark_gc_gz"],
        "segmentation": result["object_plus_action_segmentation"]["benchmark_gc_gz"],
        "recovery": result["object_plus_high_confidence_action_recovery"][
            "benchmark_gc_gz"
        ],
    }
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
