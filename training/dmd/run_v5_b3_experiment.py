"""Run the Roadwatch V5 B3 frozen-ResNet18 + temporal-GRU experiment."""

from __future__ import annotations

import argparse
import json
import platform
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
    _calibrate_threshold,
    _evaluate_predictions,
    _ground_truth,
    _serializable_prediction,
    discover_external_dmd_root,
)
from training.dmd.v5_resnet_gru import (
    EmbeddingClipDataset,
    ResNetGRUClassifier,
    SubjectEmbeddings,
    _weights_path,
    extract_resnet18_embeddings,
    predict_embeddings,
    train_resnet_gru,
)
from training.dmd.v5_video_action import (
    ACTION_CLASSES,
    classification_metrics,
    event_metrics,
    merge_evidence_events,
    probabilities_to_events,
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _find_experiment(result: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    return next(row for row in result["experiments"] if row["experiment_id"] == experiment_id)


def _recover_false_negatives(
    selected: dict[str, Any], diagnostics: dict[str, Any]
) -> list[dict[str, Any]]:
    recovered: list[dict[str, Any]] = []
    for case in diagnostics["false_negative_cases"]:
        subject_id = case["subject_id"]
        truth = case["truth"]
        subject_result = selected["per_subject"][subject_id]
        matches = subject_result.get("matches")
        if matches is None:
            matches = subject_result.get("evaluation_evidence", {}).get("matches", [])
        for match in matches:
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
    return recovered


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    b3_cfg = config["resnet_gru_action"]
    event_cfg = config["event_decision"]
    data_root = discover_external_dmd_root()
    derivatives = data_root / "derivatives"
    subjects_dir = data_root / "subjects"
    embedding_dir = derivatives / "v5_resnet18_embeddings"
    checkpoint_dir = Path("models/candidate/v5_resnet18_gru_001")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    output_dir.mkdir(parents=True, exist_ok=True)

    round1 = json.loads(
        (output_dir / "v5_experiment_results.json").read_text(encoding="utf-8")
    )
    diagnostics = json.loads(
        (output_dir / "04_false_negative_diagnostics.json").read_text(encoding="utf-8")
    )
    embeddings: dict[str, SubjectEmbeddings] = {}
    durations: dict[str, float] = {}
    embedding_provenance: list[dict[str, Any]] = []
    for subject_id in DEVELOPMENT_SUBJECTS:
        rgb_cache = derivatives / "v5_rgb_cache" / f"v5_rgb_{subject_id}_64.npz"
        output_path = embedding_dir / f"v5_resnet18_{subject_id}.npz"
        if not output_path.is_file():
            provenance = extract_resnet18_embeddings(
                subject_id, rgb_cache, output_path, batch_size=int(b3_cfg["batch_size"])
            )
        else:
            weights_path = _weights_path()
            provenance = {
                "subject_id": subject_id,
                "status": "REUSED",
                "rgb_cache_path": str(rgb_cache),
                "rgb_cache_sha256": sha256_file(rgb_cache),
                "output_path": str(output_path),
                "output_sha256": sha256_file(output_path),
                "backbone": "torchvision_resnet18_imagenet1k_v1",
                "weights_path": "torch_hub/checkpoints/resnet18-f37072fd.pth",
                "weights_sha256": sha256_file(weights_path),
            }
        embedding_provenance.append(provenance)
        embeddings[subject_id] = SubjectEmbeddings.load(subject_id, output_path)
        feature_payload = json.loads(
            (derivatives / f"v3_features_{subject_id}.json").read_text(encoding="utf-8")
        )
        durations[subject_id] = float(feature_payload["duration_seconds"])

    folds: list[dict[str, Any]] = []
    heldout_predictions: dict[str, dict[str, Any]] = {}
    for fold_index, heldout_subject in enumerate(DEVELOPMENT_SUBJECTS):
        training_subjects = tuple(
            subject_id for subject_id in DEVELOPMENT_SUBJECTS if subject_id != heldout_subject
        )
        checkpoint_path = checkpoint_dir / f"loso_{heldout_subject}.pt"
        if checkpoint_path.is_file():
            checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
            if (
                checkpoint.get("heldout_subject") != heldout_subject
                or tuple(checkpoint.get("training_subjects", ())) != training_subjects
                or checkpoint.get("architecture") != b3_cfg["architecture"]
            ):
                raise ValueError(f"checkpoint provenance mismatch: {checkpoint_path}")
            model = ResNetGRUClassifier(hidden_size=int(b3_cfg["hidden_size"]))
            model.load_state_dict(checkpoint["model_state_dict"])
            training = {
                "status": "REUSED_COMPLETED_CHECKPOINT",
                "epochs": int(b3_cfg["epochs"]),
                "training_seconds": None,
            }
        else:
            train_dataset = EmbeddingClipDataset(
                [embeddings[item] for item in training_subjects],
                sequence_length=int(b3_cfg["sequence_length"]),
                stride=int(b3_cfg["training_stride"]),
                labeled_only=True,
            )
            model, training = train_resnet_gru(
                train_dataset,
                epochs=int(b3_cfg["epochs"]),
                batch_size=int(b3_cfg["batch_size"]),
                learning_rate=float(b3_cfg["learning_rate"]),
                seed=int(b3_cfg["seed"]) + fold_index,
            )
        train_predictions = {
            subject_id: predict_embeddings(
                model,
                embeddings[subject_id],
                sequence_length=int(b3_cfg["sequence_length"]),
                batch_size=int(b3_cfg["batch_size"]),
            )
            for subject_id in training_subjects
        }
        threshold, threshold_sweep = _calibrate_threshold(
            config, train_predictions, training_subjects, subjects_dir, durations
        )
        heldout = predict_embeddings(
            model,
            embeddings[heldout_subject],
            sequence_length=int(b3_cfg["sequence_length"]),
            batch_size=int(b3_cfg["batch_size"]),
        )
        heldout_events = probabilities_to_events(
            heldout["timestamps"],
            heldout["probabilities"],
            threshold=threshold,
            release_threshold=threshold * float(event_cfg["release_ratio"]),
            minimum_samples=int(event_cfg["minimum_positive_samples"]),
            merge_gap_seconds=float(event_cfg["merge_gap_seconds"]),
        )
        heldout["events"] = heldout_events
        heldout_predictions[heldout_subject] = heldout
        checkpoint_payload = {
            "model_state_dict": model.state_dict(),
            "architecture": b3_cfg["architecture"],
            "class_names": ACTION_CLASSES,
            "heldout_subject": heldout_subject,
            "training_subjects": training_subjects,
            "action_threshold": threshold,
            "config": b3_cfg,
            "dataset_role": "DEVELOPMENT_ONLY_LOSO",
            "forbidden_consumed_holdouts": CONSUMED_HOLDOUTS,
            "backbone_weights_sha256": embedding_provenance[0]["weights_sha256"],
            "git_head": _git_head(),
        }
        torch.save(checkpoint_payload, checkpoint_path)
        heldout_event_metrics = event_metrics(
            _ground_truth(subjects_dir, heldout_subject),
            heldout_events,
            durations[heldout_subject],
        )
        heldout_event_metrics["duration_seconds"] = durations[heldout_subject]
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
                "heldout_event_metrics": heldout_event_metrics,
                "runtime": _serializable_prediction(heldout),
                "checkpoint_path": str(checkpoint_path),
                "checkpoint_sha256": sha256_file(checkpoint_path),
            }
        )

    pooled_clip = classification_metrics(
        np.concatenate([heldout_predictions[item]["labels"] for item in DEVELOPMENT_SUBJECTS]),
        np.concatenate(
            [heldout_predictions[item]["probabilities"] for item in DEVELOPMENT_SUBJECTS],
            axis=0,
        ),
    )
    action_events = {
        subject_id: heldout_predictions[subject_id]["events"]
        for subject_id in DEVELOPMENT_SUBJECTS
    }
    action_pooled, action_per_subject = _evaluate_predictions(
        subjects_dir, durations, action_events
    )
    object_experiment = _find_experiment(round1, "B0_V4_CURRENT")
    object_events = {
        subject_id: object_experiment["per_subject"][subject_id]["evaluation_evidence"][
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
                event
                for event in action_events[subject_id]
                if event["confidence"] >= 0.70 and event["duration"] >= 1.0
            ],
        )
        for subject_id in DEVELOPMENT_SUBJECTS
    }
    reliable_pooled, reliable_per_subject = _evaluate_predictions(
        subjects_dir, durations, reliable_events
    )
    new_experiments = [
        {
            "experiment_id": "B3_RESNET18_GRU",
            "name": "Frozen ImageNet ResNet-18 embeddings plus bidirectional GRU",
            "pooled_metrics": action_pooled,
            "per_subject": action_per_subject,
            "status": "MEASURED_ON_ROADWATCH_SUBJECT_DISJOINT",
        },
        {
            "experiment_id": "B8_B3_LATE_FUSION",
            "name": "Frozen V4 object events plus B3 action candidates",
            "pooled_metrics": fused_pooled,
            "per_subject": fused_per_subject,
            "status": "MEASURED_ON_ROADWATCH_SUBJECT_DISJOINT",
        },
        {
            "experiment_id": "B8_B3_RELIABILITY_FUSION",
            "name": "Reliability-gated B3 action candidates plus V4 object events",
            "pooled_metrics": reliable_pooled,
            "per_subject": reliable_per_subject,
            "status": "MEASURED_ON_ROADWATCH_SUBJECT_DISJOINT",
        },
    ]
    experiments = [*round1["experiments"], *new_experiments]
    eligible = [
        row
        for row in experiments
        if row["pooled_metrics"]["false_alarms_per_minute"]
        <= float(config["selection"]["maximum_false_alarms_per_minute"])
    ]
    selected = max(
        eligible,
        key=lambda row: (
            row["pooled_metrics"]["f1"],
            row["pooled_metrics"]["recall"],
            -row["pooled_metrics"]["false_alarms_per_minute"],
        ),
    )
    recovery_by_experiment = {
        row["experiment_id"]: _recover_false_negatives(row, diagnostics)
        for row in experiments
    }
    recovered = recovery_by_experiment[selected["experiment_id"]]

    metadata_path = checkpoint_dir / "metadata.json"
    metadata = {
        "schema_version": 1,
        "model_version": "ROADWATCH_V5_RESNET18_GRU_001",
        "architecture": b3_cfg["architecture"],
        "activation_state": "DEVELOPMENT_ONLY_NOT_VALIDATED",
        "dataset_role": "DEVELOPMENT_ONLY",
        "backbone_weights_sha256": embedding_provenance[0]["weights_sha256"],
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
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    runtime_rows = [fold["runtime"] for fold in folds]
    result = {
        **round1,
        "created_at": datetime.now(UTC).isoformat(),
        "git_head": _git_head(),
        "config_sha256": sha256_file(config_path),
        "experiment_round": 2,
        "input_provenance_b3": embedding_provenance,
        "input_set_sha256_b3": stable_json_hash(embedding_provenance),
        "backbone_source": {
            "name": "torchvision ResNet-18 IMAGENET1K_V1",
            "weights_url": "https://download.pytorch.org/models/resnet18-f37072fd.pth",
            "weights_sha256": embedding_provenance[0]["weights_sha256"],
            "frozen": True,
        },
        "clip_metrics_b3": pooled_clip,
        "folds_b3": folds,
        "experiments": experiments,
        "selected_candidate": selected["experiment_id"],
        "selected_metrics": selected["pooled_metrics"],
        "v4_false_negatives_recovered": recovered,
        "recovered_count": len(recovered),
        "v4_false_negative_recovery_by_experiment": recovery_by_experiment,
        "recovered_count_by_experiment": {
            experiment_id: len(rows)
            for experiment_id, rows in recovery_by_experiment.items()
        },
        "checkpoint_metadata_path_b3": str(metadata_path),
        "checkpoint_metadata_sha256_b3": sha256_file(metadata_path),
        "runtime_b3": {
            "scope": "TEMPORAL_HEAD_ON_PRECOMPUTED_RESNET18_EMBEDDINGS",
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
            "note": (
                "Temporal head only; frozen ResNet-18 embedding extraction is measured "
                "separately in input_provenance_b3 and is not included in clips/s."
            ),
        },
        "governance": {
            **round1["governance"],
            "selected_candidate_activation": "DEVELOPMENT_ONLY_NOT_VALIDATED",
        },
    }
    output_path = output_dir / "v5_experiment_results_round2.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Run Roadwatch V5 B3 ResNet18-GRU LOSO")
    parser.add_argument(
        "--config", type=Path, default=Path("models/model_config_v5_research.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/v5_research_upgrade")
    )
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
