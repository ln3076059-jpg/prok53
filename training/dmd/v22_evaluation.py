"""Coverage and metric invariants for V22. Stage-one inputs contain no labels."""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def confusion_metrics(matrix: list[list[int]]) -> dict[str, Any]:
    cm = np.asarray(matrix)
    if cm.ndim != 2 or cm.shape[0] != cm.shape[1] or not np.issubdtype(cm.dtype, np.integer):
        raise ValueError("confusion matrix must be a square integer matrix")
    if (cm < 0).any() or cm.sum() == 0:
        raise ValueError("invalid confusion counts")
    support = cm.sum(1)
    predicted = cm.sum(0)
    diagonal = np.diag(cm)
    precision = np.divide(diagonal, predicted, out=np.zeros(len(cm)), where=predicted > 0)
    recall = np.divide(diagonal, support, out=np.zeros(len(cm)), where=support > 0)
    f1 = np.divide(
        2 * diagonal, support + predicted, out=np.zeros(len(cm)), where=(support + predicted) > 0
    )
    return {
        "confusion_matrix": cm.tolist(),
        "sample_count": int(cm.sum()),
        "support": support.tolist(),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1": f1.tolist(),
        "macro_f1": float(f1.mean()),
        "weighted_f1": float(np.dot(f1, support) / support.sum()),
        "balanced_accuracy": float(recall.mean()),
        "zero_division": 0,
    }


def metrics_from_raw(labels: np.ndarray, predictions: np.ndarray, classes: int) -> dict:
    from sklearn.metrics import confusion_matrix, precision_recall_fscore_support

    if labels.shape != predictions.shape or labels.ndim != 1 or not len(labels):
        raise ValueError("raw label and prediction counts must agree and be nonempty")
    if not (
        np.isin(labels, np.arange(classes)).all() and np.isin(predictions, np.arange(classes)).all()
    ):
        raise ValueError("class mapping mismatch")
    cm = confusion_matrix(labels, predictions, labels=list(range(classes)))
    metrics = confusion_metrics(cm.tolist())
    _, _, f1, support = precision_recall_fscore_support(
        labels, predictions, labels=list(range(classes)), zero_division=0
    )
    if not np.allclose(metrics["f1"], f1, atol=1e-12) or metrics["support"] != support.tolist():
        raise ValueError("metrics do not reconcile with sklearn")
    return metrics


def verify_report_metrics(measured: dict, reported: dict, tolerance: float = 1e-6) -> None:
    for key in ("sample_count", "support", "confusion_matrix"):
        if key in reported and reported[key] != measured[key]:
            raise ValueError(f"INCONSISTENT_REPORT: {key}")
    if "macro_f1" in reported and not math.isclose(
        measured["macro_f1"], reported["macro_f1"], abs_tol=tolerance
    ):
        raise ValueError("INCONSISTENT_REPORT: macro_f1")


def validate_full_coverage(
    sessions: list[dict], expected_subjects: list[str], expected_gt: int
) -> None:
    if sorted(item["subject_id"] for item in sessions) != sorted(expected_subjects):
        raise ValueError("INCOMPLETE_COHORT")
    for item in sessions:
        expected_indices = range(0, item["source_frames"], item["sample_stride"])
        if item["frame_indices"] != list(expected_indices):
            raise ValueError("INCOMPLETE_FRAME_COVERAGE")
        if item.get("scope") != "FULL_COHORT_AUTONOMOUS_EXPERIMENT":
            raise ValueError("SMOKE_CANNOT_BE_FULL_BENCHMARK")
        if not math.isclose(
            item["evaluated_seconds"], item["source_frames"] / item["fps"], abs_tol=1e-6
        ):
            raise ValueError("EXPOSURE_MISMATCH")
    if sum(item["gt_count"] for item in sessions) != expected_gt:
        raise ValueError("GT_COUNT_MISMATCH")


def boundary_metrics(matches: list[dict], gt: list[dict], predictions: list[dict]) -> dict:
    """Boundary quality is separate from frozen event F1; no credit for unseen tails."""
    coverage = []
    fragments = []
    for target in gt:
        start, end = target["start_seconds"], target["end_seconds"]
        intervals = sorted(
            (max(start, p["start_seconds"]), min(end, p["end_seconds"])) for p in predictions
        )
        intervals = [(a, b) for a, b in intervals if b > a]
        fragments.append(len(intervals))
        union = 0.0
        cursor = start
        for left, right in intervals:
            union += max(0.0, right - max(left, cursor))
            cursor = max(cursor, right)
        coverage.append(union / (end - start) if end > start else 0.0)
    return {
        "temporal_iou": [m["iou"] for m in matches],
        "onset_latency_seconds": [m["start_latency"] for m in matches],
        "offset_latency_seconds": [m["end_latency"] for m in matches],
        "predicted_durations_seconds": [p["end_seconds"] - p["start_seconds"] for p in predictions],
        "gt_durations_seconds": [g["end_seconds"] - g["start_seconds"] for g in gt],
        "event_coverage": coverage,
        "fragment_count_per_gt": fragments,
        "fragmentation_rate": sum(n > 1 for n in fragments) / len(gt) if gt else None,
    }
