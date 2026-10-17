"""Run exploratory V11 domain-normalization and subject-balancing ablations."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.common import sha256_file
from training.dmd.causal_motion import (
    causal_running_standardize,
    spatial_residual_standardize,
)
from training.dmd.v11_representation_experiments import (
    ROOT,
    SEED,
    SUBJECTS,
    _evaluate,
    _load_subject,
    _metrics,
    _separability,
)

DEFAULT_OUTPUT = ROOT / "reports" / "v11_research"
PLAN = {
    "E0_RAW_CAUSAL_MULTISCALE": "Unchanged V11-C causal multiscale feature",
    "E1_SPATIAL_RESIDUAL_NORM": "Per-frame spatial residual normalization; no future context",
    "E2_CAUSAL_RUNNING_NORM": "Normalize using at most 120 earlier samples; no subject ID input",
    "E3_SPATIAL_PLUS_RUNNING": "Spatial normalization followed by past-only running normalization",
    "E4_SUBJECT_CLASS_BALANCED": "Raw causal feature with deterministic equal subject-class sampling",
}


def _balanced_indices(subjects: list[dict[str, Any]]) -> np.ndarray:
    groups = []
    for subject_index, subject in enumerate(subjects):
        for class_index in range(3):
            local = np.flatnonzero(subject["labels"] == class_index)
            groups.append((subject_index, class_index, local))
    target = min(len(local) for _, _, local in groups)
    offsets = np.cumsum([0] + [len(subject["labels"]) for subject in subjects])
    selected = []
    rng = np.random.default_rng(SEED)
    for subject_index, _, local in groups:
        chosen = np.sort(rng.choice(local, size=target, replace=False))
        selected.append(chosen + offsets[subject_index])
    return np.sort(np.concatenate(selected))


def _evaluate_balanced(
    representation_id: str,
    subjects: list[dict[str, Any]],
    output: Path,
) -> dict[str, Any]:
    folds = []
    pooled_labels = []
    pooled_predictions = []
    for heldout in subjects:
        training = [row for row in subjects if row["subject_id"] != heldout["subject_id"]]
        train_x = np.concatenate([row["features"][representation_id] for row in training])
        train_y = np.concatenate([row["labels"] for row in training])
        selected = _balanced_indices(training)
        test_x = heldout["features"][representation_id]
        test_y = heldout["labels"]
        test_mask = test_y >= 0
        model = make_pipeline(
            StandardScaler(),
            OneVsRestClassifier(
                LogisticRegression(
                    C=1.0,
                    class_weight=None,
                    max_iter=500,
                    random_state=SEED,
                    solver="liblinear",
                ),
                n_jobs=1,
            ),
        )
        started = time.perf_counter()
        model.fit(train_x[selected], train_y[selected])
        predictions = model.predict(test_x[test_mask]).astype(np.int64)
        probabilities = model.predict_proba(test_x[test_mask]).astype(np.float32)
        labels = test_y[test_mask]
        path = output / "predictions" / f"{representation_id.lower()}_loso_{heldout['subject_id']}.npz"
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            frame_indices=heldout["frame_indices"][test_mask],
            timestamps=heldout["timestamps"][test_mask],
            labels=labels,
            predictions=predictions,
            probabilities=probabilities,
        )
        row = _metrics(labels, predictions)
        row.update(
            {
                "heldout_subject": heldout["subject_id"],
                "training_subjects": [item["subject_id"] for item in training],
                "training_samples_after_balance": int(len(selected)),
                "fit_predict_seconds": round(time.perf_counter() - started, 3),
                "prediction_path": path.relative_to(ROOT).as_posix(),
                "prediction_sha256": sha256_file(path),
            }
        )
        folds.append(row)
        pooled_labels.append(labels)
        pooled_predictions.append(predictions)
    pooled = _metrics(np.concatenate(pooled_labels), np.concatenate(pooled_predictions))
    return {
        "representation_id": representation_id,
        "description": PLAN[representation_id],
        "feature_dimensions": int(subjects[0]["features"][representation_id].shape[1]),
        "folds": folds,
        "pooled": pooled,
        "worst_subject": {
            "macro_f1": min(row["macro_f1"] for row in folds),
            "safe_recall": min(row["safe_recall"] for row in folds),
            "phone_recall": min(row["phone_recall"] for row in folds),
            "other_recall": min(row["other_recall"] for row in folds),
        },
    }


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    subjects = [_load_subject(subject_id) for subject_id in SUBJECTS]
    for subject in subjects:
        raw = subject["features"]["V11-C_CAUSAL_MULTISCALE"]
        spatial = spatial_residual_standardize(raw)
        subject["features"] = {
            "E0_RAW_CAUSAL_MULTISCALE": raw,
            "E1_SPATIAL_RESIDUAL_NORM": spatial,
            "E2_CAUSAL_RUNNING_NORM": causal_running_standardize(raw),
            "E3_SPATIAL_PLUS_RUNNING": causal_running_standardize(spatial),
            "E4_SUBJECT_CLASS_BALANCED": raw,
        }
    started = time.perf_counter()
    results = []
    for representation_id in PLAN:
        if representation_id == "E4_SUBJECT_CLASS_BALANCED":
            row = _evaluate_balanced(representation_id, subjects, output)
        else:
            row = _evaluate(representation_id, subjects, output)
            row["description"] = PLAN[representation_id]
        row["separability"] = _separability(representation_id, subjects)
        results.append(row)
    baseline = results[0]
    eligible = [
        row
        for row in results[1:]
        if row["pooled"]["macro_f1"] > baseline["pooled"]["macro_f1"]
        and row["worst_subject"]["safe_recall"] >= 0.20
        and row["worst_subject"]["macro_f1"] >= 0.20
    ]
    selected = max(
        eligible,
        key=lambda row: (row["worst_subject"]["safe_recall"], row["pooled"]["macro_f1"]),
        default=None,
    )
    return {
        "schema_version": 1,
        "experiment_family": "V11_SUBJECT_GENERALIZATION",
        "evidence_level": "LEVEL_2_EXPLORATORY_REPEATED_DEVELOPMENT",
        "hypothesis": "Causal online normalization or equal subject-class sampling reduces motion-domain shift and improves worst-subject recall.",
        "pre_registered_plan": PLAN,
        "selection_rule": "Improve pooled Macro-F1 over E0 with worst-subject SAFE recall and Macro-F1 each at least 20%; rank by worst SAFE recall then pooled Macro-F1.",
        "runtime_subject_identity_input": False,
        "subjects": list(SUBJECTS),
        "protected_holdouts_accessed": False,
        "representations": results,
        "selected_candidate": selected["representation_id"] if selected else None,
        "decision": "KEEP_DEVELOPMENT_HYPOTHESIS" if selected else "REJECT_ALL_CANDIDATES",
        "promotion_decision": "REJECT",
        "runtime_seconds": round(time.perf_counter() - started, 3),
        "limitations": [
            "The same three development subjects have been examined repeatedly.",
            "Each subject cache is one continuous session; online state reset behavior needs new sessions.",
            "Group DRO is blocked because three subjects are insufficient for credible selection and validation.",
            "No Event F1 or independent holdout is available.",
        ],
        "production_approved": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = run(args.output)
    path = args.output / "generalization_experiments.json"
    path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"result": str(path), "decision": result["decision"]}, indent=2))


if __name__ == "__main__":
    main()
