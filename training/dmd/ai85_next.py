"""Data-centric diagnostics and subject-normalized linear probes for AI85 next.

The helpers in this module operate only on the three already-consumed development
subjects. They never load the protected gZ-37 or gE-28 holdouts.
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression

from training.dmd.holdout_guard import assert_not_consumed_holdout_for_development
from training.dmd.v5_resnet_gru import SubjectEmbeddings
from training.dmd.v5_video_action import event_metrics, probabilities_to_events

DEVELOPMENT_SUBJECTS = ("gB-9", "gZ-36", "gC-14")
FORBIDDEN_HOLDOUTS = ("gZ-37", "gE-28")
PHONE_LABELS = (1, 2, 3, 4)


@dataclass(frozen=True)
class EventPolicy:
    threshold: float
    minimum_samples: int
    merge_gap_seconds: float
    release_ratio: float = 0.65

    def as_dict(self) -> dict[str, float | int]:
        return {
            "threshold": self.threshold,
            "release_threshold": self.threshold * self.release_ratio,
            "minimum_positive_samples": self.minimum_samples,
            "merge_gap_seconds": self.merge_gap_seconds,
        }


def normalize_per_video(
    embeddings: np.ndarray, *, epsilon: float = 1e-4
) -> tuple[np.ndarray, dict[str, Any]]:
    """Remove per-video appearance offset using label-free batch statistics."""

    values = np.asarray(embeddings, dtype=np.float32)
    if values.ndim != 2 or not len(values):
        raise ValueError("embeddings must be a non-empty [samples, features] array")
    mean = values.mean(axis=0, dtype=np.float64).astype(np.float32)
    scale = values.std(axis=0, dtype=np.float64).astype(np.float32)
    normalized = (values - mean) / np.maximum(scale, epsilon)
    return normalized.astype(np.float32), {
        "epsilon": epsilon,
        "input_shape": list(values.shape),
        "mean_absolute_post_mean": float(np.abs(normalized.mean(axis=0)).mean()),
        "mean_post_std": float(normalized.std(axis=0).mean()),
        "constant_or_near_constant_features": int(np.sum(scale < epsilon)),
        "finite": bool(np.isfinite(normalized).all()),
    }


def load_normalized_subject(
    subject_id: str, embedding_dir: Path, *, epsilon: float = 1e-4
) -> tuple[SubjectEmbeddings, dict[str, Any]]:
    assert_not_consumed_holdout_for_development(
        subject_id, caller_action="ai85_next_subject_normalization"
    )
    source = SubjectEmbeddings.load(
        subject_id, embedding_dir / f"v5_resnet18_{subject_id}.npz"
    )
    normalized, diagnostics = normalize_per_video(source.embeddings, epsilon=epsilon)
    return (
        SubjectEmbeddings(
            subject_id=subject_id,
            embeddings=normalized,
            timestamps=source.timestamps,
            labels=source.labels,
        ),
        diagnostics,
    )


def fit_logistic_probe(
    subjects: Iterable[SubjectEmbeddings],
    *,
    task: str,
    regularization_c: float,
    seed: int,
    max_iter: int = 300,
    tolerance: float = 1e-3,
) -> LogisticRegression:
    subjects = list(subjects)
    if not subjects:
        raise ValueError("at least one training subject is required")
    features = np.concatenate(
        [subject.embeddings[subject.labels >= 0] for subject in subjects]
    )
    labels = np.concatenate([subject.labels[subject.labels >= 0] for subject in subjects])
    if task == "flat":
        targets = labels
    elif task == "coarse":
        targets = np.where(labels == 0, 0, np.where(labels == 5, 2, 1))
    elif task == "phone_binary":
        targets = np.isin(labels, PHONE_LABELS).astype(np.int64)
    else:
        raise ValueError(f"unsupported probe task: {task}")
    model = LogisticRegression(
        C=regularization_c,
        class_weight="balanced",
        max_iter=max_iter,
        random_state=seed,
        tol=tolerance,
    )
    model.fit(features, targets)
    return model


def prediction_centers(samples: int, sequence_length: int = 9) -> np.ndarray:
    offset = sequence_length // 2
    if samples <= 2 * offset:
        return np.asarray([], dtype=np.int64)
    return np.arange(offset, samples - offset, dtype=np.int64)


def predict_probe(
    model: LogisticRegression, subject: SubjectEmbeddings, *, task: str
) -> dict[str, np.ndarray]:
    centers = prediction_centers(len(subject.embeddings))
    raw = model.predict_proba(subject.embeddings[centers]).astype(np.float32)
    if task == "flat":
        probabilities = raw
        phone_probabilities = raw[:, 1:5].sum(axis=1)
    elif task == "coarse":
        probabilities = np.zeros((len(raw), 6), dtype=np.float32)
        probabilities[:, 0] = raw[:, 0]
        probabilities[:, 1] = raw[:, 1]
        probabilities[:, 5] = raw[:, 2]
        phone_probabilities = raw[:, 1]
    elif task == "phone_binary":
        phone_probabilities = raw[:, 1]
        probabilities = np.zeros((len(raw), 6), dtype=np.float32)
        probabilities[:, 0] = 1.0 - phone_probabilities
        probabilities[:, 1] = phone_probabilities
    else:
        raise ValueError(f"unsupported probe task: {task}")
    return {
        "centers": centers,
        "timestamps": subject.timestamps[centers],
        "labels": subject.labels[centers],
        "probabilities": probabilities,
        "phone_probabilities": phone_probabilities.astype(np.float32),
    }


def events_from_prediction(
    prediction: dict[str, np.ndarray], policy: EventPolicy
) -> list[dict[str, Any]]:
    return probabilities_to_events(
        prediction["timestamps"],
        prediction["probabilities"],
        threshold=policy.threshold,
        release_threshold=policy.threshold * policy.release_ratio,
        minimum_samples=policy.minimum_samples,
        merge_gap_seconds=policy.merge_gap_seconds,
    )


def aggregate_event_metrics(
    truths: dict[str, list[dict[str, Any]]],
    predictions: dict[str, list[dict[str, Any]]],
    durations: dict[str, float],
) -> tuple[dict[str, Any], dict[str, dict[str, Any]]]:
    if set(predictions) != set(truths):
        raise ValueError("prediction and truth subject sets must match")
    per_subject = {
        subject: event_metrics(truths[subject], predictions[subject], durations[subject])
        for subject in truths
    }
    tp = sum(row["true_positives"] for row in per_subject.values())
    fp = sum(row["false_positives"] for row in per_subject.values())
    fn = sum(row["false_negatives"] for row in per_subject.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    minutes = sum(durations[subject] for subject in truths) / 60.0
    pooled = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fp / max(minutes, 1e-9), 4),
        "evaluated_minutes": round(minutes, 4),
    }
    return pooled, per_subject


def longest_true_run(mask: np.ndarray) -> int:
    longest = current = 0
    for value in np.asarray(mask, dtype=bool):
        current = current + 1 if value else 0
        longest = max(longest, current)
    return longest


def interval_trace(
    timestamps: np.ndarray,
    phone_probabilities: np.ndarray,
    start_seconds: float,
    end_seconds: float,
    *,
    threshold: float,
) -> dict[str, Any]:
    mask = (timestamps >= start_seconds) & (timestamps <= end_seconds)
    local_times = timestamps[mask]
    local_scores = phone_probabilities[mask]
    if not len(local_scores):
        return {
            "samples": 0,
            "missing_centered_samples": True,
            "probability_max": None,
            "probability_mean": None,
            "fraction_at_or_above_threshold": 0.0,
            "longest_activation_run": 0,
            "first_sample_offset_seconds": None,
            "last_sample_offset_seconds": None,
        }
    active = local_scores >= threshold
    return {
        "samples": int(len(local_scores)),
        "missing_centered_samples": False,
        "probability_min": float(local_scores.min()),
        "probability_mean": float(local_scores.mean()),
        "probability_median": float(np.median(local_scores)),
        "probability_max": float(local_scores.max()),
        "probability_p90": float(np.percentile(local_scores, 90)),
        "fraction_at_or_above_threshold": float(active.mean()),
        "longest_activation_run": longest_true_run(active),
        "first_sample_offset_seconds": float(local_times[0] - start_seconds),
        "last_sample_offset_seconds": float(end_seconds - local_times[-1]),
        "trace": [
            {"timestamp_seconds": float(time), "phone_probability": float(score)}
            for time, score in zip(local_times, local_scores, strict=True)
        ],
    }


def policy_candidates() -> list[EventPolicy]:
    return [
        EventPolicy(threshold, minimum, gap)
        for threshold in (0.70, 0.74, 0.78, 0.80, 0.82, 0.84)
        for minimum in (2, 3, 4)
        for gap in (1.0, 2.0)
    ]


def policy_rank(metrics: dict[str, Any], policy: EventPolicy) -> tuple[Any, ...]:
    """Prefer the FA gate, then F1/recall/precision, then conservative policy."""

    return (
        metrics["false_alarms_per_minute"] > 0.15,
        -metrics["f1"],
        -metrics["recall"],
        -metrics["precision"],
        -policy.minimum_samples,
        -policy.threshold,
        policy.merge_gap_seconds,
    )


def interval_sample_times(start_seconds: float, end_seconds: float, samples: int) -> np.ndarray:
    if samples < 1:
        raise ValueError("samples must be positive")
    if end_seconds < start_seconds:
        raise ValueError("event end must not precede event start")
    if samples == 1 or end_seconds == start_seconds:
        return np.asarray([(start_seconds + end_seconds) / 2.0], dtype=np.float64)
    margin = min((end_seconds - start_seconds) * 0.1, 0.25)
    return np.linspace(start_seconds + margin, end_seconds - margin, samples)
