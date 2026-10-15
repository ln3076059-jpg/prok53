"""Lightweight, subject-disjoint video action recognition for Roadwatch V5.

This module deliberately implements the lowest-cost rung of the V5 research plan:
a small RGB CNN followed by temporal pooling.  It consumes short BODY-view clips and
does not use YOLO confidences or pose features, so it can provide independent action
evidence when a small phone is not detected.  The model is research-only; an action-
only positive is emitted as NEEDS_REVIEW, never as an automatic violation.
"""

from __future__ import annotations

import hashlib
import os
import random
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from training.dmd.adapter import DMDSessionAnnotation, parse_dmd_openlabel
from training.dmd.holdout_guard import assert_not_consumed_holdout_for_development
from training.dmd.temporal_eval import TemporalMatchingPolicy, match_temporal_events

ACTION_CLASSES = (
    "SAFE_DRIVING",
    "TEXTING_LEFT",
    "TEXTING_RIGHT",
    "PHONECALL_LEFT",
    "PHONECALL_RIGHT",
    "OTHER_ACTIVITY",
)
PHONE_CLASS_IDS = (1, 2, 3, 4)
UNKNOWN_CLASS = "UNKNOWN"

SOURCE_ACTION_TO_CLASS = {
    "driver_actions/safe_drive": "SAFE_DRIVING",
    "driver_actions/texting_left": "TEXTING_LEFT",
    "driver_actions/texting_right": "TEXTING_RIGHT",
    "driver_actions/phonecall_left": "PHONECALL_LEFT",
    "driver_actions/phonecall_right": "PHONECALL_RIGHT",
    "driver_actions/reach_side": "OTHER_ACTIVITY",
    "driver_actions/hair_and_makeup": "OTHER_ACTIVITY",
    "driver_actions/radio": "OTHER_ACTIVITY",
    "driver_actions/unclassified": "OTHER_ACTIVITY",
}


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def set_deterministic(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError:
        pass
    cv2.setNumThreads(1)


def label_timestamps(
    timestamps: np.ndarray,
    annotation: DMDSessionAnnotation,
) -> tuple[np.ndarray, list[str]]:
    """Map sampled timestamps to human DMD driver-action intervals.

    UNKNOWN is retained as an abstention label and excluded from supervised loss.
    Phone actions take priority if malformed source intervals overlap.
    """

    labels = np.full(len(timestamps), -1, dtype=np.int64)
    sources = [UNKNOWN_CLASS] * len(timestamps)
    class_to_id = {name: index for index, name in enumerate(ACTION_CLASSES)}
    intervals = [
        item for item in annotation.all_intervals if item.source_action in SOURCE_ACTION_TO_CLASS
    ]
    intervals.sort(
        key=lambda item: item.source_action
        not in {
            "driver_actions/texting_left",
            "driver_actions/texting_right",
            "driver_actions/phonecall_left",
            "driver_actions/phonecall_right",
        }
    )
    for interval in intervals:
        mapped = SOURCE_ACTION_TO_CLASS[interval.source_action]
        mask = (timestamps >= interval.start_seconds) & (timestamps <= interval.end_seconds)
        unassigned = mask & (labels < 0)
        labels[unassigned] = class_to_id[mapped]
        for index in np.flatnonzero(unassigned):
            sources[int(index)] = interval.source_action
    return labels, sources


def extract_rgb_cache(
    subject_id: str,
    video_path: Path,
    feature_payload: dict[str, Any],
    annotation_path: Path,
    output_path: Path,
    *,
    roi_xyxy: tuple[float, float, float, float] = (0.0, 0.10, 0.65, 0.95),
    image_size: int = 64,
) -> dict[str, Any]:
    """Extract small, aligned RGB driver crops at the existing V3 sample timestamps."""

    assert_not_consumed_holdout_for_development(subject_id, caller_action="v5_rgb_cache")
    records = feature_payload["records"]
    targets = np.asarray([int(item["frame_index"]) for item in records], dtype=np.int64)
    timestamps = np.asarray(
        [float(item["timestamp_seconds"]) for item in records], dtype=np.float32
    )
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open BODY video: {video_path}")

    frames = np.empty((len(targets), image_size, image_size, 3), dtype=np.uint8)
    current = -1
    t0 = time.perf_counter()
    for output_index, target in enumerate(targets):
        while current < int(target):
            if not cap.grab():
                cap.release()
                raise ValueError(f"video ended before frame {target}: {video_path}")
            current += 1
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            cap.release()
            raise ValueError(f"cannot decode frame {target}: {video_path}")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = roi_xyxy
        crop = frame[
            max(0, int(y1 * height)) : min(height, int(y2 * height)),
            max(0, int(x1 * width)) : min(width, int(x2 * width)),
        ]
        resized = cv2.resize(crop, (image_size, image_size), interpolation=cv2.INTER_AREA)
        frames[output_index] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    cap.release()

    annotation = parse_dmd_openlabel(annotation_path)
    labels, source_actions = label_timestamps(timestamps, annotation)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        frames=frames,
        timestamps=timestamps,
        frame_indices=targets,
        labels=labels,
    )
    return {
        "subject_id": subject_id,
        "status": "EXTRACTED",
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "video_path": str(video_path),
        "video_sha256": sha256_file(video_path),
        "annotation_path": str(annotation_path),
        "annotation_sha256": sha256_file(annotation_path),
        "records": len(frames),
        "labeled_records": int(np.sum(labels >= 0)),
        "unknown_records": int(np.sum(labels < 0)),
        "source_action_counts": {
            key: source_actions.count(key) for key in sorted(set(source_actions))
        },
        "roi_xyxy": list(roi_xyxy),
        "image_size": image_size,
        "extraction_seconds": round(time.perf_counter() - t0, 3),
    }


@dataclass(frozen=True)
class SubjectFrames:
    subject_id: str
    frames: np.ndarray
    timestamps: np.ndarray
    labels: np.ndarray

    @classmethod
    def load(cls, subject_id: str, path: Path) -> SubjectFrames:
        assert_not_consumed_holdout_for_development(subject_id, caller_action="v5_cache_load")
        payload = np.load(path, allow_pickle=False)
        return cls(
            subject_id=subject_id,
            frames=payload["frames"],
            timestamps=payload["timestamps"].astype(np.float64),
            labels=payload["labels"].astype(np.int64),
        )


def clip_indices(length: int, sequence_length: int, stride: int = 1) -> list[np.ndarray]:
    half = sequence_length // 2
    centers = range(half, max(half, length - half), stride)
    return [np.arange(center - half, center - half + sequence_length) for center in centers]


class ClipDataset(Dataset):
    def __init__(
        self,
        subjects: Iterable[SubjectFrames],
        *,
        sequence_length: int,
        stride: int,
        labeled_only: bool,
    ) -> None:
        self.subjects = list(subjects)
        self.items: list[tuple[int, np.ndarray, int]] = []
        center_offset = sequence_length // 2
        for subject_index, subject in enumerate(self.subjects):
            for indices in clip_indices(len(subject.frames), sequence_length, stride):
                center = int(indices[center_offset])
                label = int(subject.labels[center])
                if labeled_only and label < 0:
                    continue
                self.items.append((subject_index, indices, center))

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, int, int]:
        subject_index, indices, center = self.items[index]
        subject = self.subjects[subject_index]
        frames = torch.from_numpy(subject.frames[indices].copy()).permute(0, 3, 1, 2).float()
        frames = frames.div_(255.0)
        label = int(subject.labels[center])
        return frames, label, subject_index, center


class TinyTemporalCNN(nn.Module):
    """Small frame encoder with order-aware temporal pooling.

    Mean/max pooled appearance and the first-to-last feature delta retain coarse motion
    without the memory and compute cost of 3D ResNet or CLIP.
    """

    def __init__(self, num_classes: int = len(ACTION_CLASSES)) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(3, 12, 3, stride=2, padding=1),
            nn.BatchNorm2d(12),
            nn.SiLU(),
            nn.Conv2d(12, 24, 3, stride=2, padding=1),
            nn.BatchNorm2d(24),
            nn.SiLU(),
            nn.Conv2d(24, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(32 * 3, 64),
            nn.SiLU(),
            nn.Dropout(0.15),
            nn.Linear(64, num_classes),
        )

    def forward(self, clips: torch.Tensor) -> torch.Tensor:
        batch, steps, channels, height, width = clips.shape
        encoded = self.encoder(clips.reshape(batch * steps, channels, height, width))
        encoded = encoded.reshape(batch, steps, -1)
        pooled = torch.cat(
            (encoded.mean(dim=1), encoded.max(dim=1).values, encoded[:, -1] - encoded[:, 0]),
            dim=1,
        )
        return self.classifier(pooled)


def class_weights(dataset: ClipDataset) -> torch.Tensor:
    counts = np.zeros(len(ACTION_CLASSES), dtype=np.float64)
    for subject_index, _, center in dataset.items:
        label = int(dataset.subjects[subject_index].labels[center])
        if label >= 0:
            counts[label] += 1
    safe = np.maximum(counts, 1.0)
    weights = np.sqrt(safe.sum() / safe)
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


def train_model(
    dataset: ClipDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[TinyTemporalCNN, dict[str, Any]]:
    set_deterministic(seed)
    model = TinyTemporalCNN()
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    criterion = nn.CrossEntropyLoss(weight=class_weights(dataset), label_smoothing=0.03)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    model.train()
    for epoch in range(epochs):
        total_loss = 0.0
        correct = 0
        seen = 0
        for clips, labels, _, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(clips)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += len(labels)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(total_loss / max(seen, 1), 6),
                "training_accuracy": round(correct / max(seen, 1), 6),
            }
        )
    return model, {
        "epochs": epochs,
        "samples": len(dataset),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_seconds": round(time.perf_counter() - started, 3),
        "history": history,
    }


def predict_subject(
    model: TinyTemporalCNN,
    subject: SubjectFrames,
    *,
    sequence_length: int,
    batch_size: int,
) -> dict[str, Any]:
    dataset = ClipDataset([subject], sequence_length=sequence_length, stride=1, labeled_only=False)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities: list[np.ndarray] = []
    centers: list[int] = []
    per_clip_latency_ms: list[float] = []
    process = psutil.Process(os.getpid())
    rss_before_mb = process.memory_info().rss / (1024 * 1024)
    peak_rss_mb = [rss_before_mb]
    stop_memory_sampling = threading.Event()

    def sample_memory() -> None:
        while not stop_memory_sampling.wait(0.01):
            peak_rss_mb[0] = max(peak_rss_mb[0], process.memory_info().rss / (1024 * 1024))

    memory_sampler = threading.Thread(target=sample_memory, daemon=True)
    memory_sampler.start()
    started = time.perf_counter()
    model.eval()
    try:
        with torch.inference_mode():
            for clips, _, _, batch_centers in loader:
                batch_started = time.perf_counter()
                probabilities.append(torch.softmax(model(clips), dim=1).cpu().numpy())
                batch_elapsed = time.perf_counter() - batch_started
                per_clip_latency_ms.extend(
                    [1000.0 * batch_elapsed / max(len(clips), 1)] * len(clips)
                )
                centers.extend(int(value) for value in batch_centers.numpy())
    finally:
        stop_memory_sampling.set()
        memory_sampler.join()
        rss_after_mb = process.memory_info().rss / (1024 * 1024)
        peak_rss_mb[0] = max(peak_rss_mb[0], rss_after_mb)
    elapsed = time.perf_counter() - started
    probs = (
        np.concatenate(probabilities, axis=0)
        if probabilities
        else np.empty((0, len(ACTION_CLASSES)))
    )
    centers_array = np.asarray(centers, dtype=np.int64)
    return {
        "subject_id": subject.subject_id,
        "centers": centers_array,
        "timestamps": subject.timestamps[centers_array],
        "labels": subject.labels[centers_array],
        "probabilities": probs,
        "inference_seconds": round(elapsed, 6),
        "clips_per_second": round(len(dataset) / elapsed, 3) if elapsed else 0.0,
        "milliseconds_per_clip": round(1000.0 * elapsed / max(len(dataset), 1), 3),
        "p50_latency_ms": round(float(np.percentile(per_clip_latency_ms, 50)), 3)
        if per_clip_latency_ms
        else 0.0,
        "p95_latency_ms": round(float(np.percentile(per_clip_latency_ms, 95)), 3)
        if per_clip_latency_ms
        else 0.0,
        "rss_before_mb": round(rss_before_mb, 3),
        "rss_after_mb": round(rss_after_mb, 3),
        "peak_process_rss_mb": round(peak_rss_mb[0], 3),
    }


def classification_metrics(labels: np.ndarray, probabilities: np.ndarray) -> dict[str, Any]:
    valid = labels >= 0
    truth = labels[valid]
    predicted = probabilities[valid].argmax(axis=1) if np.any(valid) else np.asarray([], dtype=int)
    confusion = np.zeros((len(ACTION_CLASSES), len(ACTION_CLASSES)), dtype=np.int64)
    for actual, guess in zip(truth, predicted, strict=True):
        confusion[int(actual), int(guess)] += 1
    per_class: dict[str, dict[str, float | int]] = {}
    f1_values: list[float] = []
    for index, name in enumerate(ACTION_CLASSES):
        tp = int(confusion[index, index])
        fp = int(confusion[:, index].sum() - tp)
        fn = int(confusion[index, :].sum() - tp)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1_values.append(f1)
        per_class[name] = {
            "support": int(confusion[index, :].sum()),
            "precision": round(precision, 4),
            "recall": round(recall, 4),
            "f1": round(f1, 4),
        }
    return {
        "clip_accuracy": round(float(np.mean(truth == predicted)), 4) if len(truth) else 0.0,
        "macro_f1": round(float(np.mean(f1_values)), 4),
        "per_class": per_class,
        "confusion_matrix": confusion.tolist(),
        "class_order": list(ACTION_CLASSES),
        "evaluated_clips": int(len(truth)),
        "unknown_ground_truth_clips_excluded": int(np.sum(~valid)),
    }


def probabilities_to_events(
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    *,
    threshold: float,
    release_threshold: float | None = None,
    minimum_samples: int = 2,
    merge_gap_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    if len(timestamps) == 0:
        return []
    phone_scores = probabilities[:, PHONE_CLASS_IDS].sum(axis=1)
    if len(phone_scores) >= 3:
        phone_scores = np.convolve(phone_scores, np.ones(3) / 3.0, mode="same")
    release = release_threshold if release_threshold is not None else threshold * 0.65
    active_start: int | None = None
    positive_run = 0
    events: list[dict[str, Any]] = []
    for index, score in enumerate(phone_scores):
        positive_run = positive_run + 1 if score >= threshold else 0
        if active_start is None and positive_run >= minimum_samples:
            active_start = index - minimum_samples + 1
        if active_start is not None and score < release:
            end_index = max(active_start, index - 1)
            if end_index - active_start + 1 >= minimum_samples:
                events.append(
                    {
                        "event_type": "PHONE",
                        "start_seconds": round(float(timestamps[active_start]), 3),
                        "end_seconds": round(float(timestamps[end_index]), 3),
                        "confidence": round(
                            float(phone_scores[active_start : end_index + 1].max()), 4
                        ),
                        "source": "VIDEO_ACTION",
                        "review_status": "NEEDS_REVIEW",
                    }
                )
            active_start = None
            positive_run = 0
    if active_start is not None:
        end_index = len(timestamps) - 1
        if end_index - active_start + 1 >= minimum_samples:
            events.append(
                {
                    "event_type": "PHONE",
                    "start_seconds": round(float(timestamps[active_start]), 3),
                    "end_seconds": round(float(timestamps[end_index]), 3),
                    "confidence": round(float(phone_scores[active_start : end_index + 1].max()), 4),
                    "source": "VIDEO_ACTION",
                    "review_status": "NEEDS_REVIEW",
                }
            )
    merged: list[dict[str, Any]] = []
    for event in events:
        if merged and event["start_seconds"] <= merged[-1]["end_seconds"] + merge_gap_seconds:
            merged[-1]["end_seconds"] = event["end_seconds"]
            merged[-1]["confidence"] = max(merged[-1]["confidence"], event["confidence"])
        else:
            merged.append(dict(event))
    for event in merged:
        event["duration"] = round(event["end_seconds"] - event["start_seconds"], 3)
    return merged


def event_metrics(
    ground_truth: list[dict[str, Any]],
    predictions: list[dict[str, Any]],
    duration_seconds: float,
) -> dict[str, Any]:
    matched, unmatched_predictions, unmatched_ground_truth = match_temporal_events(
        ground_truth, predictions, TemporalMatchingPolicy()
    )
    distinct_matches = {
        (
            float(item["ground_truth"]["start_seconds"]),
            float(item["ground_truth"]["end_seconds"]),
        )
        for item in matched
    }
    tp = len(distinct_matches)
    duplicates = max(0, len(matched) - tp)
    fp = len(unmatched_predictions) + duplicates
    fn = len(unmatched_ground_truth)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    latencies = [float(item["start_latency"]) for item in matched]
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fp / max(duration_seconds / 60.0, 1e-9), 4),
        "detection_latency_mean_seconds": round(float(np.mean(latencies)), 3)
        if latencies
        else None,
        "duplicate_events": duplicates,
        "event_fragmentation": duplicates,
        "predicted_events": predictions,
        "matches": matched,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_ground_truth": unmatched_ground_truth,
    }


def merge_evidence_events(
    object_events: list[dict[str, Any]],
    action_events: list[dict[str, Any]],
    *,
    gap_seconds: float = 2.0,
) -> list[dict[str, Any]]:
    """Late fusion with provenance and conservative review status."""

    combined = []
    for item in object_events:
        event = dict(item)
        event["source"] = "OBJECT_GEOMETRY"
        event["review_status"] = "PENDING"
        combined.append(event)
    combined.extend(dict(item) for item in action_events)
    combined.sort(key=lambda item: float(item["start_seconds"]))
    merged: list[dict[str, Any]] = []
    for event in combined:
        if merged and event["start_seconds"] <= merged[-1]["end_seconds"] + gap_seconds:
            previous = merged[-1]
            sources = set(previous.get("sources", [previous.get("source", "UNKNOWN")]))
            sources.update(event.get("sources", [event.get("source", "UNKNOWN")]))
            previous["end_seconds"] = max(previous["end_seconds"], event["end_seconds"])
            previous["confidence"] = max(
                previous.get("confidence", 0.0), event.get("confidence", 0.0)
            )
            previous["sources"] = sorted(sources)
            previous["review_status"] = (
                "PENDING" if "OBJECT_GEOMETRY" in sources else "NEEDS_REVIEW"
            )
        else:
            new_event = dict(event)
            new_event["sources"] = [new_event.pop("source", "UNKNOWN")]
            merged.append(new_event)
    for event in merged:
        event["duration"] = round(event["end_seconds"] - event["start_seconds"], 3)
    return merged
