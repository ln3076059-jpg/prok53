"""Train one V7 LOSO fold in an isolated process."""

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
from training.dmd.v5_resnet_gru import EmbeddingClipDataset, SubjectEmbeddings
from training.dmd.v5_video_action import probabilities_to_events
from training.dmd.v7_action_models import (
    mine_training_hard_negatives,
    predict_v7,
    train_binary_gate_v7,
    train_hierarchical_v7,
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
        raise ValueError(f"invalid heldout subject: {heldout_subject}")
    variant = variants[variant_id]
    backbone = config["backbone"]
    event_cfg = config["event_decision"]
    embedding_dir = discover_external_dmd_root() / "derivatives" / "v5_resnet18_embeddings"
    training_subjects = tuple(
        subject for subject in DEVELOPMENT_SUBJECTS if subject != heldout_subject
    )
    subjects = {
        subject: SubjectEmbeddings.load(
            subject, embedding_dir / f"v5_resnet18_{subject}.npz"
        )
        for subject in (*training_subjects, heldout_subject)
    }
    sequence_length = int(variant["sequence_length"])
    train_dataset = EmbeddingClipDataset(
        [subjects[subject] for subject in training_subjects],
        sequence_length=sequence_length,
        stride=int(backbone["training_stride"]),
        labeled_only=True,
    )
    fold_index = DEVELOPMENT_SUBJECTS.index(heldout_subject)
    seed = int(backbone["seed"]) + fold_index
    hard_negatives: set[tuple[int, int]] = set()
    mining: dict[str, Any] = {"enabled": False, "count": 0}
    if bool(variant.get("hard_negative_mining", False)):
        miner_rows: list[dict[str, Any]] = []
        for target_index, target_subject in enumerate(training_subjects):
            source_subject = next(
                subject for subject in training_subjects if subject != target_subject
            )
            source_dataset = EmbeddingClipDataset(
                [subjects[source_subject]],
                sequence_length=sequence_length,
                stride=int(backbone["training_stride"]),
                labeled_only=True,
            )
            target_dataset = EmbeddingClipDataset(
                [subjects[target_subject]],
                sequence_length=sequence_length,
                stride=int(backbone["training_stride"]),
                labeled_only=True,
            )
            miner, miner_training = train_hierarchical_v7(
                source_dataset,
                epochs=int(variant["miner_epochs"]),
                batch_size=int(backbone["batch_size"]),
                learning_rate=float(backbone["learning_rate"]),
                seed=seed + 10_000 + target_index,
                coarse_loss_mode=variant["coarse_loss_mode"],
                fine_loss_mode=variant["fine_loss_mode"],
                coarse_loss_coefficient=float(variant["coarse_loss_coefficient"]),
                fine_loss_coefficient=float(variant["fine_loss_coefficient"]),
            )
            target_hard_negatives = mine_training_hard_negatives(
                miner,
                target_dataset,
                batch_size=int(backbone["batch_size"]),
                minimum_phone_probability=float(
                    variant["miner_minimum_phone_probability"]
                ),
            )
            remapped = {
                (target_index, center) for _, center in target_hard_negatives
            }
            hard_negatives.update(remapped)
            miner_rows.append(
                {
                    "source_subject": source_subject,
                    "target_subject": target_subject,
                    "count": len(remapped),
                    "training": miner_training,
                }
            )
            del miner
        mining = {
            "enabled": True,
            "scope": (
                "TWO_WAY_CROSS_SUBJECT_WITHIN_TRAINING_POOL; "
                "VERIFIED_GROUND_TRUTH_SAFE_OR_OTHER"
            ),
            "outer_heldout_accessed": False,
            "count": len(hard_negatives),
            "minimum_phone_probability": variant["miner_minimum_phone_probability"],
            "inner_folds": miner_rows,
        }
    if variant["architecture"] == "hierarchical":
        model, training = train_hierarchical_v7(
            train_dataset,
            epochs=int(backbone["epochs"]),
            batch_size=int(backbone["batch_size"]),
            learning_rate=float(backbone["learning_rate"]),
            seed=seed,
            coarse_loss_mode=variant["coarse_loss_mode"],
            fine_loss_mode=variant["fine_loss_mode"],
            coarse_loss_coefficient=float(variant["coarse_loss_coefficient"]),
            fine_loss_coefficient=float(variant["fine_loss_coefficient"]),
            hard_negative_centers=hard_negatives,
            hard_negative_weight=float(variant.get("hard_negative_weight", 1.0)),
        )
    elif variant["architecture"] == "binary_gate":
        model, training = train_binary_gate_v7(
            train_dataset,
            epochs=int(backbone["epochs"]),
            batch_size=int(backbone["batch_size"]),
            learning_rate=float(backbone["learning_rate"]),
            seed=seed,
        )
    else:
        raise ValueError(f"unsupported architecture: {variant['architecture']}")
    heldout_dataset = EmbeddingClipDataset(
        [subjects[heldout_subject]],
        sequence_length=sequence_length,
        stride=1,
        labeled_only=False,
    )
    prediction = predict_v7(
        model,
        heldout_dataset,
        architecture=variant["architecture"],
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
    checkpoint_dir = Path("models/candidate/v7_research")
    fold_dir = Path("reports/v7_optimization/folds")
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
            "mining": mining,
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
        "mining": mining,
        "checkpoint_path": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "prediction_path": str(prediction_path),
        "prediction_sha256": sha256_file(prediction_path),
        "fixed_action_threshold": threshold,
        "inference_seconds": prediction["inference_seconds"],
        "clips_per_second": prediction["clips_per_second"],
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
