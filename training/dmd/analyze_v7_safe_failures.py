"""Analyze V6 hierarchical SAFE failures without fitting to outer-fold labels."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS
from training.dmd.v5_video_action import ACTION_CLASSES
from training.dmd.v6_action_hierarchy import COARSE_CLASSES


def _quantiles(values: np.ndarray) -> dict[str, float | None]:
    if not len(values):
        return {str(q): None for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)}
    return {
        str(q): round(float(np.quantile(values, q)), 6)
        for q in (0.0, 0.1, 0.25, 0.5, 0.75, 0.9, 1.0)
    }


def _confusion(truth: np.ndarray, predicted: np.ndarray, classes: int) -> list[list[int]]:
    matrix = np.zeros((classes, classes), dtype=np.int64)
    for expected, actual in zip(truth, predicted, strict=True):
        matrix[int(expected), int(actual)] += 1
    return matrix.tolist()


def _runs(values: np.ndarray, timestamps: np.ndarray) -> dict[str, Any]:
    if not len(values):
        return {"runs": 0, "transition_rate": 0.0, "per_class": {}}
    starts = np.r_[0, np.flatnonzero(values[1:] != values[:-1]) + 1]
    ends = np.r_[starts[1:] - 1, len(values) - 1]
    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.0
    per_class: dict[str, Any] = {}
    for class_index, class_name in enumerate(COARSE_CLASSES):
        durations = np.asarray(
            [
                timestamps[end] - timestamps[start] + step
                for start, end in zip(starts, ends, strict=True)
                if values[start] == class_index
            ],
            dtype=np.float64,
        )
        per_class[class_name] = {
            "runs": int(len(durations)),
            "duration_seconds_quantiles": _quantiles(durations),
        }
    return {
        "runs": int(len(starts)),
        "transition_rate": round(float((len(starts) - 1) / max(len(values) - 1, 1)), 6),
        "per_class": per_class,
    }


def analyze_subject(prediction_path: Path) -> dict[str, Any]:
    with np.load(prediction_path, allow_pickle=False) as payload:
        timestamps = payload["timestamps"]
        labels = payload["labels"]
        probabilities = payload["probabilities"]
    valid = labels >= 0
    labels = labels[valid]
    timestamps = timestamps[valid]
    probabilities = probabilities[valid]
    coarse_truth = np.where(labels == 0, 0, np.where(labels == 5, 2, 1))
    coarse_probabilities = np.column_stack(
        (probabilities[:, 0], probabilities[:, 1:5].sum(axis=1), probabilities[:, 5])
    )
    coarse_predicted = coarse_probabilities.argmax(axis=1)
    flat_predicted = probabilities.argmax(axis=1)
    safe = coarse_truth == 0
    phone_probability = coarse_probabilities[:, 1]
    negative_probability = np.maximum(coarse_probabilities[:, 0], coarse_probabilities[:, 2])
    margin = phone_probability - negative_probability
    phone_fine = probabilities[:, 1:5] / np.maximum(phone_probability[:, None], 1e-12)
    safe_phone = safe & (coarse_predicted == 1)
    safe_other = safe & (coarse_predicted == 2)
    return {
        "clips": int(len(labels)),
        "class_distribution": {
            name: int(np.sum(labels == index)) for index, name in enumerate(ACTION_CLASSES)
        },
        "coarse_class_distribution": {
            name: int(np.sum(coarse_truth == index))
            for index, name in enumerate(COARSE_CLASSES)
        },
        "coarse_confusion_matrix": _confusion(
            coarse_truth, coarse_predicted, len(COARSE_CLASSES)
        ),
        "flat_confusion_matrix": _confusion(labels, flat_predicted, len(ACTION_CLASSES)),
        "safe": {
            "support": int(np.sum(safe)),
            "correct": int(np.sum(safe & (coarse_predicted == 0))),
            "predicted_phone": int(np.sum(safe_phone)),
            "predicted_other": int(np.sum(safe_other)),
            "recall": round(float(np.mean(coarse_predicted[safe] == 0)), 6)
            if np.any(safe)
            else None,
            "coarse_probability_quantiles": {
                name: _quantiles(coarse_probabilities[safe, index])
                for index, name in enumerate(COARSE_CLASSES)
            },
            "phone_vs_negative_margin_quantiles": _quantiles(margin[safe]),
            "misclassified_as_phone_margin_quantiles": _quantiles(margin[safe_phone]),
            "misclassified_as_other_phone_margin_quantiles": _quantiles(margin[safe_other]),
            "phone_fine_argmax_on_phone_errors": {
                ACTION_CLASSES[index + 1]: int(
                    np.sum(phone_fine[safe_phone].argmax(axis=1) == index)
                )
                for index in range(4)
            }
            if np.any(safe_phone)
            else {name: 0 for name in ACTION_CLASSES[1:5]},
        },
        "temporal_stability": _runs(coarse_predicted, timestamps),
    }


def run(v6_dir: Path, output_path: Path) -> dict[str, Any]:
    per_subject = {
        subject_id: analyze_subject(
            v6_dir / "folds" / f"hierarchical_loss_only_loso_{subject_id}.npz"
        )
        for subject_id in DEVELOPMENT_SUBJECTS
    }
    pooled_matrices = np.sum(
        [row["coarse_confusion_matrix"] for row in per_subject.values()], axis=0
    )
    safe_support = int(pooled_matrices[0].sum())
    v6_result = json.loads(
        (v6_dir / "v6_class_collapse_ablation.json").read_text(encoding="utf-8")
    )
    fp_audit = v6_result["false_positive_audits"][
        "V6_HIERARCHICAL_LOSS_ONLY_FUSION"
    ]
    fp_durations = np.asarray(
        [
            row["prediction"]["end_seconds"] - row["prediction"]["start_seconds"]
            for row in fp_audit
        ],
        dtype=np.float64,
    )
    fp_categories: dict[str, int] = {}
    for row in fp_audit:
        category = row["primary_category"]
        fp_categories[category] = fp_categories.get(category, 0) + 1
    result = {
        "schema_version": 1,
        "source_variant": "V6_HIERARCHICAL_LOSS_ONLY",
        "source_predictions": "OUTER_LOSO_HELDOUT_ONLY",
        "coarse_class_order": list(COARSE_CLASSES),
        "flat_class_order": list(ACTION_CLASSES),
        "per_subject": per_subject,
        "pooled": {
            "coarse_confusion_matrix": pooled_matrices.tolist(),
            "safe_support": safe_support,
            "safe_recall": round(float(pooled_matrices[0, 0] / max(safe_support, 1)), 6),
            "safe_to_phone": int(pooled_matrices[0, 1]),
            "safe_to_other": int(pooled_matrices[0, 2]),
        },
        "fused_false_positive_audit": {
            "count": len(fp_audit),
            "categories": fp_categories,
            "duration_seconds_quantiles": _quantiles(fp_durations),
            "cases": fp_audit,
        },
        "interpretation": [
            "SAFE collapse is primarily SAFE-to-OTHER, not SAFE-to-PHONE.",
            "Phone fine-head changes alone cannot recover SAFE clips rejected by the coarse head.",
            "All diagnostics use held-out outer-fold predictions; no threshold is selected here.",
        ],
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--v6-dir", type=Path, default=Path("reports/v6_completion"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v7_optimization/v7_safe_failure_analysis.json"),
    )
    args = parser.parse_args()
    result = run(args.v6_dir, args.output)
    print(json.dumps(result["pooled"], indent=2))


if __name__ == "__main__":
    main()
