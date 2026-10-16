"""Verify AI85 next-stage artifacts, predictions, hashes, and preservation guarantees."""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from training.common import sha256_file
from training.dmd.ai85_audit import read_json, write_json
from training.dmd.ai85_next import load_normalized_subject, prediction_centers
from training.dmd.run_v5_experiments import discover_external_dmd_root

REQUIRED_REPORTS = (
    "ROOT_CAUSE_ANALYSIS.md",
    "SAFE_CLASSIFICATION_ANALYSIS.md",
    "FN_RECOVERY.md",
    "VISUAL_REPRESENTATION_ABLATION.md",
    "DATASET_GENERALIZATION.md",
    "METRICS_COMPARISON.md",
    "TEST_RESULTS.md",
    "FINAL_REPORT.md",
)


def artifact(path: Path) -> dict[str, Any]:
    return {
        "path": path.as_posix(),
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - values.max(axis=1, keepdims=True)
    exponential = np.exp(shifted)
    return exponential / exponential.sum(axis=1, keepdims=True)


def _predict_model(path: Path, features: np.ndarray) -> tuple[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        coefficients = payload["coefficients"].astype(np.float32)
        intercept = payload["intercept"].astype(np.float32)
        task = str(payload["task"].item())
    logits = features @ coefficients.T + intercept
    if logits.shape[1] == 1:
        positive = 1.0 / (1.0 + np.exp(-logits[:, 0]))
        raw = np.column_stack((1.0 - positive, positive))
    else:
        raw = _softmax(logits)
    if task == "flat":
        probabilities = raw
    elif task == "coarse":
        probabilities = np.zeros((len(raw), 6), dtype=np.float32)
        probabilities[:, 0] = raw[:, 0]
        probabilities[:, 1] = raw[:, 1]
        probabilities[:, 5] = raw[:, 2]
    elif task == "phone_binary":
        probabilities = raw[:, 1]
    else:
        raise ValueError(f"unsupported saved task: {task}")
    return task, probabilities.astype(np.float32)


def verify(output: Path) -> dict[str, Any]:
    analysis = read_json(output / "analysis.json")
    tests = read_json(output / "test_results.json")
    detector = read_json(output / "detector_trace.json")
    dataset = read_json(output / "dataset_generalization.json")
    root = discover_external_dmd_root()
    embedding_dir = root / "derivatives/v5_resnet18_embeddings"
    errors = []

    for name in REQUIRED_REPORTS:
        path = output / name
        if not path.is_file() or not path.read_text(encoding="utf-8").strip():
            errors.append(f"missing or empty required report: {name}")
    if analysis["selection"]["selected_candidate"] is not None:
        errors.append("a candidate was selected without meeting the governed target")
    if analysis["selection"]["production_approved"]:
        errors.append("production approval must remain false")
    if dataset["holdouts_read"]:
        errors.append("a protected holdout was read")
    if detector["cases"] != 13:
        errors.append("detector trace does not cover all 13 baseline false negatives")
    if tests["targeted"]["counts"]["failures"] or tests["targeted"]["counts"]["errors"]:
        errors.append("targeted tests did not pass")

    fold_checks = []
    for fold in analysis["folds"]:
        heldout = fold["heldout_subject"]
        subject, _ = load_normalized_subject(heldout, embedding_dir)
        centers = prediction_centers(len(subject.embeddings))
        prediction_path = Path(fold["prediction"]["path"])
        if sha256_file(prediction_path) != fold["prediction"]["sha256"]:
            errors.append(f"prediction hash mismatch: {heldout}")
        with np.load(prediction_path, allow_pickle=False) as prediction:
            if not np.array_equal(prediction["centers"], centers):
                errors.append(f"prediction centers mismatch: {heldout}")
            expected = {
                "flat": prediction["flat_probabilities"],
                "coarse": prediction["coarse_probabilities"],
                "phone_binary": prediction["phone_probabilities"],
            }
            max_errors = {}
            for name, model_row in fold["models"].items():
                model_path = Path(model_row["path"])
                if sha256_file(model_path) != model_row["sha256"]:
                    errors.append(f"model hash mismatch: {heldout}/{name}")
                task, actual = _predict_model(model_path, subject.embeddings[centers])
                difference = float(np.max(np.abs(actual - expected[task])))
                max_errors[task] = difference
                if difference > 2e-6:
                    errors.append(
                        f"prediction replay mismatch: {heldout}/{task} max_error={difference}"
                    )
        fold_checks.append({"heldout_subject": heldout, "max_replay_errors": max_errors})

    old_report_changes = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-only",
            "--",
            "reports/ai85_upgrade",
            "reports/v6_completion",
            "reports/v7_optimization",
        ],
        text=True,
    ).splitlines()
    if old_report_changes:
        errors.append(f"historical reports changed: {old_report_changes}")

    preservation = read_json("reports/ai85_upgrade/environment_audit.json")
    active_checks = []
    for row in preservation["active_models"]:
        path = Path(row["path"])
        current = sha256_file(path)
        active_checks.append({"path": row["path"], "expected": row["sha256"], "actual": current})
        if current != row["sha256"]:
            errors.append(f"active model changed: {path}")

    source_paths = [
        Path("models/model_config_ai85_next.yaml"),
        Path("training/dmd/ai85_next.py"),
        Path("training/dmd/run_ai85_next.py"),
        Path("training/dmd/ai85_next_detector_trace.py"),
        Path("training/dmd/render_ai85_next_reports.py"),
        Path("training/dmd/verify_ai85_next.py"),
        Path("tests/test_ai85_next.py"),
    ]
    model_paths = [
        Path(row["path"])
        for fold in analysis["folds"]
        for row in fold["models"].values()
    ]
    output_paths = sorted(
        path
        for path in output.rglob("*")
        if path.is_file() and path.name not in {"artifact_manifest.json", "validation.json"}
    )
    manifest = {
        "source": [artifact(path) for path in source_paths],
        "models": [artifact(path) for path in sorted(model_paths)],
        "reports_and_predictions": [artifact(path) for path in output_paths],
    }
    write_json(output / "artifact_manifest.json", manifest)
    result = {
        "status": "PASS" if not errors else "FAIL",
        "errors": errors,
        "required_reports": list(REQUIRED_REPORTS),
        "fold_checks": fold_checks,
        "historical_report_changes": old_report_changes,
        "active_model_checks": active_checks,
        "manifest": artifact(output / "artifact_manifest.json"),
    }
    write_json(output / "validation.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/ai85_next"))
    args = parser.parse_args()
    result = verify(args.output)
    print(json.dumps(result, indent=2))
    if result["status"] != "PASS":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
