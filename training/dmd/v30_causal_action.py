"""Causal six-class action head for local-only DMD development.

The input is a frozen per-frame ResNet-18 embedding.  Every training and
inference window ends at its target timestamp, and the GRU is unidirectional;
therefore no future sample can affect the current prediction.
"""

from __future__ import annotations

import time
from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.dmd.temporal_eval import TemporalMatchingPolicy, match_temporal_events
from training.dmd.v5_resnet_gru import SubjectEmbeddings
from training.dmd.v5_video_action import ACTION_CLASSES, PHONE_CLASS_IDS, set_deterministic


CLASS_TO_SOURCE_ACTION = {
    1: "driver_actions/texting_left",
    2: "driver_actions/texting_right",
    3: "driver_actions/phonecall_left",
    4: "driver_actions/phonecall_right",
}
SOURCE_ACTION_TO_CLASS = {value: key for key, value in CLASS_TO_SOURCE_ACTION.items()}


def causal_clip_indices(length: int, sequence_length: int, stride: int = 1) -> list[np.ndarray]:
    """Return past-only windows whose final index is the prediction target."""

    if sequence_length < 1 or stride < 1:
        raise ValueError("sequence_length and stride must be positive")
    return [
        np.arange(end - sequence_length + 1, end + 1)
        for end in range(sequence_length - 1, length, stride)
    ]


class CausalEmbeddingDataset(Dataset):
    def __init__(
        self,
        subjects: Iterable[SubjectEmbeddings],
        *,
        sequence_length: int,
        stride: int,
        labeled_only: bool,
        num_classes: int = len(ACTION_CLASSES),
    ) -> None:
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.num_classes = num_classes
        self.subjects = list(subjects)
        self.items: list[tuple[int, np.ndarray, int]] = []
        for subject_index, subject in enumerate(self.subjects):
            for indices in causal_clip_indices(len(subject.embeddings), sequence_length, stride):
                target = int(indices[-1])
                label = int(subject.labels[target])
                if labeled_only and label < 0:
                    continue
                if label >= num_classes:
                    raise ValueError(
                        f"label {label} is outside configured num_classes={num_classes}"
                    )
                self.items.append((subject_index, indices, target))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        subject_index, indices, target = self.items[index]
        subject = self.subjects[subject_index]
        sequence = torch.from_numpy(subject.embeddings[indices].copy()).float()
        return sequence, int(subject.labels[target]), subject_index, target

    def loss_weights(self) -> torch.Tensor:
        labels = np.asarray(
            [int(self.subjects[s].labels[target]) for s, _, target in self.items],
            dtype=np.int64,
        )
        counts = np.bincount(labels, minlength=self.num_classes).astype(np.float64)
        weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
        weights /= weights.mean()
        return torch.tensor(weights, dtype=torch.float32)


class CausalActionGRU(nn.Module):
    """Unidirectional temporal head with only prefix-derived pooling."""

    def __init__(
        self,
        embedding_dim: int = 512,
        hidden_size: int = 64,
        num_classes: int = len(ACTION_CLASSES),
    ) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("num_classes must be at least 2")
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=False,
        )
        self.classifier = nn.Sequential(
            nn.LayerNorm(hidden_size * 4),
            nn.Linear(hidden_size * 4, 128),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.gru(self.input_norm(sequences))
        pooled = torch.cat(
            (
                encoded[:, -1],
                encoded.mean(dim=1),
                encoded.max(dim=1).values,
                encoded[:, -1] - encoded[:, 0],
            ),
            dim=1,
        )
        return self.classifier(pooled)


def train_causal_action_model(
    dataset: CausalEmbeddingDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    hidden_size: int,
    seed: int,
    weighted_loss: bool,
    num_classes: int = len(ACTION_CLASSES),
) -> tuple[CausalActionGRU, dict[str, Any]]:
    set_deterministic(seed)
    if dataset.num_classes != num_classes:
        raise ValueError("dataset and model num_classes must match")
    model = CausalActionGRU(hidden_size=hidden_size, num_classes=num_classes)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    criterion = nn.CrossEntropyLoss(
        weight=dataset.loss_weights() if weighted_loss else None,
        label_smoothing=0.03,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    history: list[dict[str, float | int]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for sequences, labels, _, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(sequences)
            loss = criterion(logits, labels)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            correct += int((logits.argmax(1) == labels).sum().item())
            seen += len(labels)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(total_loss / max(seen, 1), 6),
                "training_accuracy": round(correct / max(seen, 1), 6),
            }
        )
    return model, {
        "samples": len(dataset),
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "weighted_loss": weighted_loss,
        "training_seconds": round(time.perf_counter() - started, 3),
        "history": history,
    }


def predict_causal_action(
    model: CausalActionGRU,
    subject: SubjectEmbeddings,
    *,
    sequence_length: int,
    batch_size: int,
) -> dict[str, np.ndarray | float | str]:
    dataset = CausalEmbeddingDataset(
        [subject], sequence_length=sequence_length, stride=1, labeled_only=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities: list[np.ndarray] = []
    targets: list[int] = []
    started = time.perf_counter()
    model.eval()
    with torch.inference_mode():
        for sequences, _, _, batch_targets in loader:
            probabilities.append(torch.softmax(model(sequences), dim=1).cpu().numpy())
            targets.extend(int(value) for value in batch_targets.numpy())
    elapsed = time.perf_counter() - started
    target_array = np.asarray(targets, dtype=np.int64)
    return {
        "subject_id": subject.subject_id,
        "targets": target_array,
        "timestamps": subject.timestamps[target_array],
        "labels": subject.labels[target_array],
        "probabilities": np.concatenate(probabilities, axis=0),
        "inference_seconds": round(elapsed, 6),
        "clips_per_second": round(len(dataset) / max(elapsed, 1e-9), 3),
    }


@dataclass(frozen=True)
class ActionDecoderConfig:
    threshold: float = 0.45
    release_ratio: float = 0.70
    release_patience: int = 2
    minimum_positive_samples: int = 1
    merge_gap_seconds: float = 0.6


def decode_action_events(
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    config: ActionDecoderConfig = ActionDecoderConfig(),
) -> list[dict[str, Any]]:
    """Decode class-specific causal events with explicit transition boundaries."""

    if len(timestamps) == 0:
        return []
    if len(timestamps) != len(probabilities):
        raise ValueError("timestamps and probabilities must have equal length")
    step = float(np.median(np.diff(timestamps))) if len(timestamps) > 1 else 0.5
    active_class: int | None = None
    active_start = 0
    active_scores: list[float] = []
    positive_samples = 0
    negative_run = 0
    events: list[dict[str, Any]] = []

    def close(end_index: int) -> None:
        nonlocal active_class, active_scores, positive_samples, negative_run
        if active_class is not None and positive_samples >= config.minimum_positive_samples:
            start = float(timestamps[active_start])
            end = float(timestamps[min(end_index, len(timestamps) - 1)])
            if end <= start:
                end = start + step
            events.append(
                {
                    "event_type": "PHONE",
                    "source_action": CLASS_TO_SOURCE_ACTION[active_class],
                    "start_seconds": round(start, 4),
                    "end_seconds": round(end, 4),
                    "duration": round(end - start, 4),
                    "confidence": round(max(active_scores), 4),
                    "source": "V30_CAUSAL_ACTION",
                }
            )
        active_class = None
        active_scores = []
        positive_samples = 0
        negative_run = 0

    for index, row in enumerate(probabilities):
        phone_index = int(PHONE_CLASS_IDS[int(np.argmax(row[list(PHONE_CLASS_IDS)]))])
        phone_score = float(row[phone_index])
        predicted = int(np.argmax(row))
        candidate = phone_index if predicted == phone_index and phone_score >= config.threshold else None
        if active_class is None:
            if candidate is not None:
                active_class = candidate
                active_start = index
                active_scores = [phone_score]
                positive_samples = 1
            continue
        if candidate == active_class:
            active_scores.append(phone_score)
            positive_samples += 1
            negative_run = 0
            continue
        if candidate is not None and candidate != active_class:
            close(index)
            active_class = candidate
            active_start = index
            active_scores = [phone_score]
            positive_samples = 1
            continue
        release_score = float(row[active_class])
        if release_score >= config.threshold * config.release_ratio:
            active_scores.append(release_score)
            negative_run = 0
        else:
            negative_run += 1
            if negative_run >= config.release_patience:
                close(index)
    if active_class is not None:
        close(len(timestamps) - 1)

    merged: list[dict[str, Any]] = []
    for event in events:
        if (
            merged
            and merged[-1]["source_action"] == event["source_action"]
            and event["start_seconds"] <= merged[-1]["end_seconds"] + config.merge_gap_seconds
        ):
            merged[-1]["end_seconds"] = event["end_seconds"]
            merged[-1]["duration"] = round(
                merged[-1]["end_seconds"] - merged[-1]["start_seconds"], 4
            )
            merged[-1]["confidence"] = max(
                merged[-1]["confidence"], event["confidence"]
            )
        else:
            merged.append(dict(event))
    return merged


def class_aware_event_metrics(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    *,
    duration_seconds: float,
    policy: TemporalMatchingPolicy = TemporalMatchingPolicy(),
) -> dict[str, Any]:
    """Apply the frozen temporal policy, but never match different action labels."""

    matched: list[dict[str, Any]] = []
    unmatched_predictions: list[dict[str, Any]] = []
    unmatched_ground_truth: list[dict[str, Any]] = []
    for source_action in CLASS_TO_SOURCE_ACTION.values():
        local_gt = [row for row in ground_truth if row.get("source_action") == source_action]
        local_pred = [
            row for row in predictions if row.get("source_action") == source_action
        ]
        local_matched, local_unmatched_pred, local_unmatched_gt = match_temporal_events(
            local_gt, local_pred, policy
        )
        matched.extend(local_matched)
        unmatched_predictions.extend(local_unmatched_pred)
        unmatched_ground_truth.extend(local_unmatched_gt)
    distinct_gt = {
        (
            row["ground_truth"]["source_action"],
            float(row["ground_truth"]["start_seconds"]),
            float(row["ground_truth"]["end_seconds"]),
        )
        for row in matched
    }
    tp = len(distinct_gt)
    duplicates = max(0, len(matched) - tp)
    fp = len(unmatched_predictions) + duplicates
    fn = len(unmatched_ground_truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fp / max(duration_seconds / 60.0, 1e-9), 4),
        "duplicate_events": duplicates,
        "matched_events": matched,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_ground_truth": unmatched_ground_truth,
    }
