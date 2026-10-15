"""Train one V6 LOSO fold in an isolated process to bound CPU RAM usage."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from training.common import sha256_file
from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS
from training.dmd.run_v5_experiments import CONSUMED_HOLDOUTS, discover_external_dmd_root
from training.dmd.v5_resnet_gru import (
    EmbeddingClipDataset,
    SubjectEmbeddings,
    predict_embeddings,
    train_resnet_gru,
)
from training.dmd.v5_video_action import probabilities_to_events
from training.dmd.v6_action_hierarchy import (
    predict_hierarchical,
    train_hierarchical_resnet_gru,
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def run_fold(config_path: Path, variant_id: str, heldout_subject: str) -> dict[str, Any]:
    torch.set_num_threads(1)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    variants = {row["id"]: row for row in config["ablation_variants"]}
    if variant_id not in variants:
        raise ValueError(f"unknown variant: {variant_id}")
    if heldout_subject not in DEVELOPMENT_SUBJECTS:
        raise ValueError(f"heldout subject is not in the development pool: {heldout_subject}")

    variant = variants[variant_id]
    backbone = config["backbone"]
    event_cfg = config["event_decision"]
    embedding_dir = discover_external_dmd_root() / "derivatives" / "v5_resnet18_embeddings"
    training_subjects = tuple(
        subject_id for subject_id in DEVELOPMENT_SUBJECTS if subject_id != heldout_subject
    )
    subjects = {
        subject_id: SubjectEmbeddings.load(
            subject_id, embedding_dir / f"v5_resnet18_{subject_id}.npz"
        )
        for subject_id in (*training_subjects, heldout_subject)
    }
    train_dataset = EmbeddingClipDataset(
        [subjects[item] for item in training_subjects],
        sequence_length=int(backbone["sequence_length"]),
        stride=int(backbone["training_stride"]),
        labeled_only=True,
    )
    variant_index = list(variants).index(variant_id)
    fold_index = DEVELOPMENT_SUBJECTS.index(heldout_subject)
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
            [subjects[heldout_subject]],
            sequence_length=int(backbone["sequence_length"]),
            stride=1,
            labeled_only=False,
        )
        prediction = predict_hierarchical(
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
        prediction = predict_embeddings(
            model,
            subjects[heldout_subject],
            sequence_length=int(backbone["sequence_length"]),
            batch_size=int(backbone["batch_size"]),
        )

    threshold = float(event_cfg["fixed_action_threshold"])
    events = probabilities_to_events(
        prediction["timestamps"],
        prediction["probabilities"],
        threshold=threshold,
        release_threshold=threshold * float(event_cfg["release_ratio"]),
        minimum_samples=int(event_cfg["minimum_positive_samples"]),
        merge_gap_seconds=float(event_cfg["merge_gap_seconds"]),
    )
    checkpoint_dir = Path("models/candidate/v6_action_ablation_001")
    fold_dir = Path("reports/v6_completion/folds")
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fold_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{variant_id.lower()}_loso_{heldout_subject}"
    checkpoint_path = checkpoint_dir / f"{stem}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "variant": variant,
            "heldout_subject": heldout_subject,
            "training_subjects": training_subjects,
            "training": training,
            "action_threshold": threshold,
            "threshold_provenance": event_cfg["threshold_provenance"],
            "dataset_role": config["dataset_role"],
            "forbidden_consumed_holdouts": CONSUMED_HOLDOUTS,
            "git_head": _git_head(),
        },
        checkpoint_path,
    )
    prediction_path = fold_dir / f"{stem}.npz"
    np.savez_compressed(
        prediction_path,
        centers=prediction["centers"],
        timestamps=prediction["timestamps"],
        labels=prediction["labels"],
        probabilities=prediction["probabilities"],
    )
    result = {
        "variant_id": variant_id,
        "heldout_subject": heldout_subject,
        "training_subjects": list(training_subjects),
        "training": training,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "fixed_action_threshold": threshold,
        "events": events,
        "resource_control": "ISOLATED_PROCESS; torch_threads=1; batch_size=16",
    }
    result_path = fold_dir / f"{stem}.json"
    result_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--heldout", required=True)
    args = parser.parse_args()
    result = run_fold(args.config, args.variant, args.heldout)
    print(json.dumps({"status": "COMPLETED", **result}, indent=2))


if __name__ == "__main__":
    main()
