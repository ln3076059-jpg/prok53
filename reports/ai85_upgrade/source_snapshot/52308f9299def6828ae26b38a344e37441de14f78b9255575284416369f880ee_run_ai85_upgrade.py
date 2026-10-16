"""Auditable CPU continuation experiments; no production activation or holdout tuning."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from training.common import sha256_file
from training.dmd.ai85_action import (
    AttentionResNetGRUClassifier,
    finetune,
    load_inherited_state,
    negative_evidence_fusion,
)
from training.dmd.ai85_audit import (
    artifact,
    audit_dataset,
    audit_environment,
    audit_errors,
    baseline_inputs,
    export_fn_frames,
    read_json,
    write_json,
)
from training.dmd.holdout_guard import assert_not_consumed_holdout_for_development
from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS
from training.dmd.run_v5_experiments import (
    CONSUMED_HOLDOUTS,
    _evaluate_predictions,
    discover_external_dmd_root,
)
from training.dmd.run_v6_class_collapse_ablation import _find_experiment
from training.dmd.run_v7_experiments import _evaluate_variant
from training.dmd.v5_resnet_gru import EmbeddingClipDataset, SubjectEmbeddings
from training.dmd.v5_video_action import merge_evidence_events, probabilities_to_events
from training.dmd.v6_action_hierarchy import (
    HierarchicalResNetGRUClassifier,
    predict_hierarchical,
)

DEFAULT_CONFIG = Path("models/model_config_ai85_upgrade.yaml")
DEFAULT_OUTPUT = Path("reports/ai85_upgrade")


def validate_pool(config: dict[str, Any], subject: str) -> None:
    assert_not_consumed_holdout_for_development(subject, caller_action="ai85_finetune")
    if tuple(config["development_subjects"]) != DEVELOPMENT_SUBJECTS:
        raise ValueError("this experiment is registered for the existing development pool only")
    if subject not in DEVELOPMENT_SUBJECTS:
        raise ValueError(f"unregistered development subject: {subject}")


def run_fold(config_path: Path, output: Path, variant_id: str, heldout: str) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    validate_pool(config, heldout)
    variant = next(v for v in config["ablation_variants"] if v["id"] == variant_id)
    backbone = config["backbone"]
    torch.set_num_threads(1)
    source = Path(backbone["source_checkpoint_dir"]) / (
        backbone["source_checkpoint_stem"] + heldout + ".pt"
    )
    checkpoint = torch.load(source, map_location="cpu", weights_only=True)
    training_subjects = [s for s in DEVELOPMENT_SUBJECTS if s != heldout]
    if checkpoint["heldout_subject"] != heldout or set(
        checkpoint["training_subjects"]
    ) != set(training_subjects):
        raise ValueError("inherited checkpoint must exclude the same outer subject")
    cls = {"mean": HierarchicalResNetGRUClassifier,
           "attention": AttentionResNetGRUClassifier}[variant["pooling"]]
    model = cls(embedding_dim=backbone["embedding_dim"], hidden_size=backbone["hidden_size"])
    load_inherited_state(model, checkpoint["model_state_dict"])
    embedding_dir = Path(variant.get("embedding_dir", str(
        discover_external_dmd_root() / "derivatives/v5_resnet18_embeddings"
    )))
    subjects = [SubjectEmbeddings.load(s, embedding_dir / f"v5_resnet18_{s}.npz")
                for s in training_subjects]
    dataset = EmbeddingClipDataset(
        subjects, sequence_length=backbone["sequence_length"],
        stride=backbone["training_stride"], labeled_only=True,
    )
    training = finetune(
        model, dataset, epochs=backbone["epochs"], batch_size=backbone["batch_size"],
        learning_rate=backbone["learning_rate"],
        seed=backbone["seed"] + DEVELOPMENT_SUBJECTS.index(heldout),
    )
    # The outer cache is loaded only after optimization is complete.
    outer = SubjectEmbeddings.load(heldout, embedding_dir / f"v5_resnet18_{heldout}.npz")
    evaluation = EmbeddingClipDataset(
        [outer], sequence_length=backbone["sequence_length"], stride=1, labeled_only=False,
    )
    prediction = predict_hierarchical(model, evaluation, batch_size=backbone["batch_size"])
    cfg = config["event_decision"]
    events = probabilities_to_events(
        prediction["timestamps"], prediction["probabilities"],
        threshold=cfg["fixed_action_threshold"],
        release_threshold=cfg["fixed_action_threshold"] * cfg["release_ratio"],
        minimum_samples=cfg["minimum_positive_samples"],
        merge_gap_seconds=cfg["merge_gap_seconds"],
    )
    fold_dir = output / "folds"
    model_dir = Path("models/candidate/ai85_upgrade")
    fold_dir.mkdir(exist_ok=True, parents=True)
    model_dir.mkdir(exist_ok=True, parents=True)
    stem = f"{variant_id.lower()}_loso_{heldout}"
    model_path = model_dir / f"{stem}.pt"
    if model_path.exists() or (fold_dir / f"{stem}.json").exists():
        raise FileExistsError("refusing to overwrite an existing experiment fold")
    sources = [Path(__file__), Path("training/dmd/ai85_action.py")]
    metadata = {
        "variant_id": variant_id, "pooling": variant["pooling"], "heldout_subject": heldout,
        "training_subjects": training_subjects, "training": training,
        "inherited_checkpoint": artifact(source), "config": artifact(config_path),
        "source_code": [artifact(p) for p in sources],
        "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
        "forbidden_consumed_holdouts": list(CONSUMED_HOLDOUTS),
        "outer_labels_used_for_training_or_selection": False,
    }
    torch.save({"model_state_dict": model.state_dict(), **metadata}, model_path)
    prediction_path = fold_dir / f"{stem}.npz"
    np.savez_compressed(prediction_path, **{
        key: prediction[key] for key in
        ("centers", "timestamps", "labels", "probabilities", "coarse_probabilities")
    })
    result = {
        **metadata, "checkpoint": artifact(model_path), "prediction": artifact(prediction_path),
        "events": events,
        "inference": {key: value for key, value in prediction.items() if key not in
                      {"centers", "timestamps", "labels", "probabilities", "coarse_probabilities"}},
        "centered_clip_lookahead_seconds": float(np.median(np.diff(outer.timestamps))) *
                                           (backbone["sequence_length"] // 2),
    }
    write_json(fold_dir / f"{stem}.json", result)
    return result


def _negative_experiment(
    predictions: dict[str, Any], object_events: dict[str, Any], durations: dict[str, float],
    fusion_cfg: dict[str, Any], experiment_id: str,
) -> dict[str, Any]:
    events, audit = {}, {}
    for subject in DEVELOPMENT_SUBJECTS:
        p = predictions[subject]
        retained, audit[subject] = negative_evidence_fusion(
            object_events[subject], p["timestamps"], p["probabilities"],
            minimum_safe_probability=fusion_cfg["negative_evidence_minimum"],
            minimum_fraction=fusion_cfg["negative_evidence_fraction"],
        )
        events[subject] = merge_evidence_events(retained, p["events"])
    pooled, per_subject = _evaluate_predictions(
        discover_external_dmd_root() / "subjects", durations, events,
    )
    return {"experiment_id": experiment_id, "kind": "NEGATIVE_EVIDENCE_FUSION",
            "pooled_metrics": pooled, "per_subject": per_subject, "object_veto_audit": audit}


def assemble(config_path: Path, output: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    baseline, baseline_predictions, durations = baseline_inputs()
    object_baseline = _find_experiment(read_json(
        "reports/v5_research_upgrade/v5_experiment_results_round2.json"), "B0_V4_CURRENT")
    objects = {s: object_baseline["per_subject"][s]["evaluation_evidence"]["predicted_events"]
               for s in DEVELOPMENT_SUBJECTS}
    baseline["experiments"].append(_negative_experiment(
        baseline_predictions, objects, durations, config["fusion"], "BASELINE_NEGATIVE_FUSION"))
    variants = [baseline]
    for variant in config["ablation_variants"]:
        predictions = {}
        for subject in DEVELOPMENT_SUBJECTS:
            stem = output / "folds" / f"{variant['id'].lower()}_loso_{subject}"
            fold = read_json(stem.with_suffix(".json"))
            if fold["config"]["sha256"] != sha256_file(config_path):
                raise ValueError("fold config hash differs from registered experiment")
            with np.load(stem.with_suffix(".npz"), allow_pickle=False) as p:
                predictions[subject] = {key: p[key].copy() for key in p.files}
            predictions[subject].update(events=fold["events"], fold=fold)
        result = _evaluate_variant(
            variant, predictions, subjects_dir=discover_external_dmd_root() / "subjects",
            durations=durations, object_events=objects,
        )
        result["experiments"].append(_negative_experiment(
            predictions, objects, durations, config["fusion"], f"{variant['id']}_NEGATIVE_FUSION"))
        variants.append(result)
    rows = []
    for variant in variants:
        safe = variant["coarse_metrics"]["per_class"]["SAFE_DRIVING"]["recall"]
        for experiment in variant["experiments"]:
            metric = experiment["pooled_metrics"]
            gates = {name: metric[name] >= config["selection"][f"minimum_{name}"]
                     for name in ("precision", "recall", "f1")}
            gates.update(
                false_alarms=metric["false_alarms_per_minute"] <=
                config["selection"]["maximum_false_alarms_per_minute"],
                safe_recall=safe >= config["selection"]["minimum_safe_driving_recall"],
                action_macro_f1=variant["clip_metrics"]["macro_f1"] >=
                config["selection"]["minimum_action_macro_f1"],
                independent_holdout=False,
            )
            rows.append({"experiment_id": experiment["experiment_id"], **metric,
                         "safe_driving_recall": safe,
                         "action_macro_f1": variant["clip_metrics"]["macro_f1"],
                         "worst_subject_fa": max(p["false_alarms_per_minute"]
                                                 for p in experiment["per_subject"].values()),
                         "gates": gates, "eligible": all(gates.values())})
    result = {
        "created_at_utc": datetime.now(UTC).isoformat(), "config": artifact(config_path),
        "dataset_role": config["dataset_role"], "variants": variants, "comparison": rows,
        "selection": None, "production_approved": False, "holdouts_accessed": [],
        "limitations": ["Three repeatedly used development subjects; no independent claim.",
                        "Frozen 64-pixel RGB-derived embeddings limit spatial recovery.",
                        "Centered BiGRU clips use future context; latency is offline event onset.",
                        "Single seed per outer fold; architectural effects remain exploratory."],
    }
    write_json(output / "metrics.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--stage", choices=("audit", "train", "fold", "assemble", "all"),
                        default="all")
    parser.add_argument("--variant")
    parser.add_argument("--heldout")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.stage == "fold":
        result = run_fold(args.config, args.output, args.variant, args.heldout)
        print(json.dumps({"status": "COMPLETED", "variant": args.variant,
                          "subject": args.heldout, "training_seconds":
                          result["training"]["training_seconds"]}), flush=True)
        return
    if args.stage in ("audit", "all"):
        for label, func in (("environment", audit_environment), ("dataset", audit_dataset),
                            ("errors", audit_errors)):
            func(args.output)
            print(f"audit completed: {label}", flush=True)
        export_fn_frames(args.output, read_json(args.output / "error_analysis.json"))
        print("original FN ROI export completed", flush=True)
    if args.stage in ("train", "all"):
        started = time.perf_counter()
        config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
        for variant in config["ablation_variants"]:
            for subject in DEVELOPMENT_SUBJECTS:
                subprocess.run([
                    sys.executable, "-m", "training.dmd.run_ai85_upgrade", "--stage", "fold",
                    "--config", str(args.config), "--output", str(args.output),
                    "--variant", variant["id"], "--heldout", subject,
                ], check=True)
        write_json(args.output / "training_wall_time.json", {
            "seconds": time.perf_counter() - started,
        })
    if args.stage in ("train", "assemble", "all"):
        result = assemble(args.config, args.output)
        print(json.dumps(result["comparison"], indent=2), flush=True)


if __name__ == "__main__":
    main()
