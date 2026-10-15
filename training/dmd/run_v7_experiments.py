"""Assemble governed V7 LOSO ablations and apply unchanged selection gates."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from training.common import sha256_file, stable_json_hash
from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS
from training.dmd.run_v5_experiments import (
    CONSUMED_HOLDOUTS,
    _evaluate_predictions,
    discover_external_dmd_root,
)
from training.dmd.run_v6_class_collapse_ablation import _coarse_metrics, _find_experiment
from training.dmd.v5_video_action import (
    classification_metrics,
    merge_evidence_events,
    probabilities_to_events,
)


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def _load_predictions(fold_dir: Path, variant_id: str) -> dict[str, dict[str, Any]]:
    predictions: dict[str, dict[str, Any]] = {}
    for subject in DEVELOPMENT_SUBJECTS:
        stem = f"{variant_id.lower()}_loso_{subject}"
        result = json.loads((fold_dir / f"{stem}.json").read_text(encoding="utf-8"))
        with np.load(fold_dir / f"{stem}.npz", allow_pickle=False) as payload:
            predictions[subject] = {
                "timestamps": payload["timestamps"],
                "labels": payload["labels"],
                "probabilities": payload["probabilities"],
                "events": result["events"],
                "fold": {key: value for key, value in result.items() if key != "events"},
            }
    return predictions


def _apply_negative_margin(
    predictions: dict[str, dict[str, Any]], margin: float, event_cfg: dict[str, Any]
) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    transformed: dict[str, dict[str, Any]] = {}
    suppressed = 0
    total = 0
    for subject, row in predictions.items():
        probabilities = row["probabilities"].copy()
        phone = probabilities[:, 1:5].sum(axis=1)
        negative = np.maximum(probabilities[:, 0], probabilities[:, 5])
        mask = (phone > negative) & ((phone - negative) < margin)
        suppressed += int(mask.sum())
        total += len(mask)
        probabilities[mask, 1:5] = 0.0
        probabilities[mask] /= np.maximum(probabilities[mask].sum(axis=1, keepdims=True), 1e-12)
        threshold = float(event_cfg["fixed_action_threshold"])
        events = probabilities_to_events(
            row["timestamps"],
            probabilities,
            threshold=threshold,
            release_threshold=threshold * float(event_cfg["release_ratio"]),
            minimum_samples=int(event_cfg["minimum_positive_samples"]),
            merge_gap_seconds=float(event_cfg["merge_gap_seconds"]),
        )
        transformed[subject] = {**row, "probabilities": probabilities, "events": events}
    return transformed, {
        "fixed_margin": margin,
        "suppressed_ambiguous_phone_clips": suppressed,
        "evaluated_clips": total,
        "suppression_rate": round(suppressed / max(total, 1), 6),
        "abstention_accounting": (
            "Suppressed clips remain in all clip and event denominators; "
            "false negatives are not hidden."
        ),
    }


def _evaluate_variant(
    variant: dict[str, Any],
    predictions: dict[str, dict[str, Any]],
    *,
    subjects_dir: Path,
    durations: dict[str, float],
    object_events: dict[str, list[dict[str, Any]]],
) -> dict[str, Any]:
    labels = np.concatenate([predictions[subject]["labels"] for subject in DEVELOPMENT_SUBJECTS])
    probabilities = np.concatenate(
        [predictions[subject]["probabilities"] for subject in DEVELOPMENT_SUBJECTS], axis=0
    )
    action_events = {subject: predictions[subject]["events"] for subject in DEVELOPMENT_SUBJECTS}
    action_pooled, action_subject = _evaluate_predictions(subjects_dir, durations, action_events)
    fused_events = {
        subject: merge_evidence_events(object_events[subject], action_events[subject])
        for subject in DEVELOPMENT_SUBJECTS
    }
    fused_pooled, fused_subject = _evaluate_predictions(subjects_dir, durations, fused_events)
    return {
        "variant": variant,
        "clip_metrics": classification_metrics(labels, probabilities),
        "coarse_metrics": _coarse_metrics(labels, probabilities),
        "folds": [predictions[subject]["fold"] for subject in DEVELOPMENT_SUBJECTS],
        "experiments": [
            {
                "experiment_id": f"{variant['id']}_ACTION",
                "kind": "ACTION_ONLY",
                "pooled_metrics": action_pooled,
                "per_subject": action_subject,
            },
            {
                "experiment_id": f"{variant['id']}_FUSION",
                "kind": "V4_OBJECT_PLUS_ACTION",
                "pooled_metrics": fused_pooled,
                "per_subject": fused_subject,
            },
        ],
    }


def run(config_path: Path, output_dir: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    output_dir.mkdir(parents=True, exist_ok=True)
    data_root = discover_external_dmd_root()
    derivatives = data_root / "derivatives"
    subjects_dir = data_root / "subjects"
    durations = {
        subject: float(
            json.loads(
                (derivatives / f"v3_features_{subject}.json").read_text(encoding="utf-8")
            )["duration_seconds"]
        )
        for subject in DEVELOPMENT_SUBJECTS
    }
    v5 = json.loads(
        Path("reports/v5_research_upgrade/v5_experiment_results_round2.json").read_text(
            encoding="utf-8"
        )
    )
    object_experiment = _find_experiment(v5, "B0_V4_CURRENT")
    object_events = {
        subject: object_experiment["per_subject"][subject]["evaluation_evidence"][
            "predicted_events"
        ]
        for subject in DEVELOPMENT_SUBJECTS
    }
    v6 = json.loads(
        Path("reports/v6_completion/v6_class_collapse_ablation.json").read_text(
            encoding="utf-8"
        )
    )
    v6_baseline = next(
        row for row in v6["variants"] if row["variant"]["id"] == "HIERARCHICAL_LOSS_ONLY"
    )
    variants: list[dict[str, Any]] = [
        {
            **v6_baseline,
            "variant": {
                "id": "V7_A_V6_BASELINE",
                "architecture": "hierarchical",
                "source": "REUSED_V6_OUTER_LOSO_PREDICTIONS",
                "hypothesis": "Frozen comparator; no V7 retraining.",
            },
            "experiments": [
                {**row, "experiment_id": row["experiment_id"].replace(
                    "V6_HIERARCHICAL_LOSS_ONLY", "V7_A_V6_BASELINE"
                )}
                for row in v6_baseline["experiments"]
            ],
        }
    ]
    prediction_sets: dict[str, dict[str, dict[str, Any]]] = {}
    for variant in config["ablation_variants"]:
        predictions = _load_predictions(output_dir / "folds", variant["id"])
        prediction_sets[variant["id"]] = predictions
        variants.append(
            _evaluate_variant(
                variant,
                predictions,
                subjects_dir=subjects_dir,
                durations=durations,
                object_events=object_events,
            )
        )
    base_predictions = prediction_sets["V7_B_COARSE_OBJECTIVE"]
    margin_predictions, margin_audit = _apply_negative_margin(
        base_predictions,
        float(config["event_decision"]["fixed_negative_evidence_margin"]),
        config["event_decision"],
    )
    margin_variant = {
        "id": "V7_F_NEGATIVE_MARGIN",
        "architecture": "V7_B_POSTPROCESSOR",
        "source": "V7_B_COARSE_OBJECTIVE",
        "margin_audit": margin_audit,
        "hypothesis": "Fixed negative-evidence margin suppresses ambiguous phone activation.",
    }
    variants.append(
        _evaluate_variant(
            margin_variant,
            margin_predictions,
            subjects_dir=subjects_dir,
            durations=durations,
            object_events=object_events,
        )
    )
    selection_rows: list[dict[str, Any]] = []
    for variant_result in variants:
        safe_recall = variant_result["coarse_metrics"]["per_class"]["SAFE_DRIVING"]["recall"]
        for experiment in variant_result["experiments"]:
            worst_fa = max(
                row["false_alarms_per_minute"]
                for row in experiment["per_subject"].values()
            )
            metrics = experiment["pooled_metrics"]
            gates = {
                "aggregate_fa": metrics["false_alarms_per_minute"]
                <= float(config["selection"]["maximum_false_alarms_per_minute"]),
                "worst_subject_fa": worst_fa
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
                    "worst_subject_false_alarms_per_minute": round(worst_fa, 4),
                    "gates": gates,
                    "eligible": all(gates.values()),
                }
            )
    eligible = [row for row in selection_rows if row["eligible"]]
    selected = max(
        eligible,
        key=lambda row: (
            row["pooled_metrics"]["f1"],
            row["pooled_metrics"]["recall"],
            -row["pooled_metrics"]["false_alarms_per_minute"],
        ),
        default=None,
    )
    result = {
        "schema_version": 1,
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
            "OUTER_LOSO; fixed threshold 0.80 and margin 0.15; no per-variant outer-label tuning; "
            "iterative development on the same three subjects"
        ),
        "variants": variants,
        "selection_rule": config["selection"],
        "selection_rows": selection_rows,
        "selected_candidate": selected,
        "result_input_hash": stable_json_hash(
            {
                subject: sha256_file(
                    derivatives / "v5_resnet18_embeddings" / f"v5_resnet18_{subject}.npz"
                )
                for subject in DEVELOPMENT_SUBJECTS
            }
        ),
        "governance": {
            "activation_state": "DEVELOPMENT_ONLY_NOT_VALIDATED",
            "production_approved": False,
            "automatic_violation": False,
            "production_changed": False,
        },
    }
    output_path = output_dir / "v7_experiments.json"
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    metadata = {
        "schema_version": 1,
        "model_family": "ROADWATCH_V7_RESEARCH",
        "activation_state": "DEVELOPMENT_ONLY_NOT_VALIDATED",
        "production_approved": False,
        "selected_candidate": selected,
        "forbidden_consumed_holdouts": list(CONSUMED_HOLDOUTS),
    }
    metadata_path = Path("models/candidate/v7_research/metadata.json")
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("models/model_config_v7_optimization.yaml")
    )
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/v7_optimization")
    )
    args = parser.parse_args()
    result = run(args.config, args.output_dir)
    print(
        json.dumps(
            {"status": result["status"], "selected_candidate": result["selected_candidate"]},
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
