from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from backend.ai.fusion import FEATURE_NAMES
from training.common import sha256_file, stable_json_hash


def _load(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"split", "label", *FEATURE_NAMES}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"fusion CSV is missing columns: {sorted(missing)}")
    return rows


def _matrix(rows: list[dict[str, str]]) -> tuple[np.ndarray, np.ndarray]:
    x = np.asarray([[float(row[name]) for name in FEATURE_NAMES] for row in rows], dtype=np.float64)
    y = np.asarray([int(row["label"]) for row in rows], dtype=np.float64)
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("fusion data contains NaN/Inf")
    if not set(np.unique(y)).issubset({0.0, 1.0}):
        raise ValueError("fusion labels must be 0 or 1")
    return x, y


def _sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, -60, 60)
    return 1.0 / (1.0 + np.exp(-values))


def fit_balanced_logistic(
    x: np.ndarray,
    y: np.ndarray,
    learning_rate: float = 0.03,
    epochs: int = 3000,
    l2: float = 1e-3,
) -> tuple[np.ndarray, float, np.ndarray, np.ndarray, list[float]]:
    if len(x) < 20 or len(np.unique(y)) != 2:
        raise ValueError("fusion training needs at least 20 rows and both labels")
    means = x.mean(axis=0)
    scales = x.std(axis=0)
    scales[scales < 1e-12] = 1.0
    z = (x - means) / scales
    positives = max(float(y.sum()), 1.0)
    negatives = max(float(len(y) - y.sum()), 1.0)
    sample_weights = np.where(y == 1, len(y) / (2 * positives), len(y) / (2 * negatives))
    weights = np.zeros(z.shape[1], dtype=np.float64)
    intercept = 0.0
    losses: list[float] = []
    for epoch in range(epochs):
        probabilities = _sigmoid(z @ weights + intercept)
        error = (probabilities - y) * sample_weights
        weights -= learning_rate * ((z.T @ error) / len(y) + l2 * weights)
        intercept -= learning_rate * float(error.mean())
        if epoch % 100 == 0 or epoch == epochs - 1:
            clipped = np.clip(probabilities, 1e-9, 1 - 1e-9)
            loss = -float(
                np.mean(sample_weights * (y * np.log(clipped) + (1 - y) * np.log(1 - clipped)))
            ) + float(l2 * np.square(weights).sum() / 2)
            if not np.isfinite(loss):
                raise FloatingPointError("fusion optimization became non-finite")
            losses.append(loss)
    return weights, intercept, means, scales, losses


def binary_metrics(
    labels: np.ndarray, scores: np.ndarray, threshold: float
) -> dict[str, float | int]:
    predicted = scores >= threshold
    truth = labels == 1
    tp = int(np.logical_and(predicted, truth).sum())
    fp = int(np.logical_and(predicted, ~truth).sum())
    fn = int(np.logical_and(~predicted, truth).sum())
    tn = int(np.logical_and(~predicted, ~truth).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "threshold": float(threshold),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
    }


def choose_threshold(
    labels: np.ndarray, scores: np.ndarray, minimum_recall: float = 0.75
) -> dict[str, float | int]:
    candidates = sorted({0.0, 1.0, *[float(value) for value in scores]})
    reports = [binary_metrics(labels, scores, threshold) for threshold in candidates]
    eligible = [item for item in reports if float(item["recall"]) >= minimum_recall]
    pool = eligible or reports
    return max(
        pool,
        key=lambda item: (
            float(item["f1"]),
            float(item["precision"]),
            float(item["recall"]),
            float(item["threshold"]),
        ),
    )


def train(
    csv_path: Path,
    output: Path,
    target: str,
    development_manifest: Path,
    minimum_recall: float = 0.75,
    learning_rate: float = 0.03,
    epochs: int = 3000,
) -> dict:
    if not development_manifest.is_file():
        raise ValueError(f"development manifest is missing: {development_manifest}")
    rows = _load(csv_path)
    train_rows = [row for row in rows if row["split"].lower() == "train"]
    val_rows = [row for row in rows if row["split"].lower() == "val"]
    if not val_rows:
        raise ValueError("fusion calibration requires a separate val split")
    x_train, y_train = _matrix(train_rows)
    x_val, y_val = _matrix(val_rows)
    weights, intercept, means, scales, losses = fit_balanced_logistic(
        x_train, y_train, learning_rate=learning_rate, epochs=epochs
    )
    val_scores = _sigmoid(((x_val - means) / scales) @ weights + intercept)
    threshold_report = choose_threshold(y_val, val_scores, minimum_recall)
    artifact = {
        "schema_version": 2,
        "status": "CALIBRATED_ON_DEVELOPMENT_TRAIN_VALIDATION",
        "created_at": datetime.now(UTC).isoformat(),
        "model_type": "BALANCED_LOGISTIC_REGRESSION",
        "target": target,
        "feature_names": list(FEATURE_NAMES),
        "feature_schema_sha256": stable_json_hash(list(FEATURE_NAMES)),
        "development_manifest": {
            "path": str(development_manifest),
            "sha256": sha256_file(development_manifest),
        },
        "feature_csv": {"path": str(csv_path), "sha256": sha256_file(csv_path)},
        "means": {name: float(means[index]) for index, name in enumerate(FEATURE_NAMES)},
        "scales": {name: float(scales[index]) for index, name in enumerate(FEATURE_NAMES)},
        "weights": {name: float(weights[index]) for index, name in enumerate(FEATURE_NAMES)},
        "intercept": float(intercept),
        "threshold": threshold_report["threshold"],
        "minimum_recall": minimum_recall,
        "train_rows": len(train_rows),
        "val_rows": len(val_rows),
        "validation": threshold_report,
        "optimization": {
            "epochs": epochs,
            "learning_rate": learning_rate,
            "loss_trace_every_100_epochs": losses,
        },
        "test_rows_used": 0,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(artifact, indent=2, sort_keys=True), encoding="utf-8")
    return artifact


def main() -> None:
    parser = argparse.ArgumentParser(description="Train dependency-free V2 logistic fusion")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--development-manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("models/fusion/no_seatbelt_v2.json"))
    parser.add_argument("--target", default="NO_SEATBELT")
    parser.add_argument("--minimum-recall", type=float, default=0.75)
    parser.add_argument("--learning-rate", type=float, default=0.03)
    parser.add_argument("--epochs", type=int, default=3000)
    args = parser.parse_args()
    print(
        json.dumps(
            train(
                args.csv,
                args.output,
                args.target,
                args.development_manifest,
                args.minimum_recall,
                args.learning_rate,
                args.epochs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
