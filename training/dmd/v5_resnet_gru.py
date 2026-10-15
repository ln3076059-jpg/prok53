"""Frozen ImageNet ResNet-18 embeddings with a lightweight temporal GRU.

This is the B3 rung of the Roadwatch V5 research plan.  The ImageNet backbone is
never fine-tuned on the three-subject development pool.  Only a small temporal
head is trained, which keeps CPU cost and overfitting risk bounded.
"""

from __future__ import annotations

import os
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import psutil
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision.models import ResNet18_Weights, resnet18

from training.common import sha256_file
from training.dmd.holdout_guard import assert_not_consumed_holdout_for_development
from training.dmd.v5_video_action import (
    ACTION_CLASSES,
    SubjectFrames,
    clip_indices,
    set_deterministic,
)


@dataclass(frozen=True)
class SubjectEmbeddings:
    subject_id: str
    embeddings: np.ndarray
    timestamps: np.ndarray
    labels: np.ndarray

    @classmethod
    def load(cls, subject_id: str, path: Path) -> SubjectEmbeddings:
        assert_not_consumed_holdout_for_development(
            subject_id, caller_action="v5_resnet_embedding_load"
        )
        payload = np.load(path, allow_pickle=False)
        return cls(
            subject_id=subject_id,
            embeddings=payload["embeddings"].astype(np.float32),
            timestamps=payload["timestamps"].astype(np.float64),
            labels=payload["labels"].astype(np.int64),
        )


def _weights_path() -> Path:
    filename = Path(ResNet18_Weights.DEFAULT.url).name
    return Path(torch.hub.get_dir()) / "checkpoints" / filename


def extract_resnet18_embeddings(
    subject_id: str,
    rgb_cache_path: Path,
    output_path: Path,
    *,
    batch_size: int = 128,
) -> dict[str, Any]:
    """Encode every aligned RGB sample once with frozen ImageNet ResNet-18."""

    assert_not_consumed_holdout_for_development(
        subject_id, caller_action="v5_resnet_embedding_extract"
    )
    subject = SubjectFrames.load(subject_id, rgb_cache_path)
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    encoded: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(subject.frames), batch_size):
            frames = torch.from_numpy(subject.frames[start : start + batch_size].copy())
            frames = frames.permute(0, 3, 1, 2).float().div_(255.0)
            frames = torch.nn.functional.interpolate(
                frames, size=(112, 112), mode="bilinear", align_corners=False
            )
            frames = (frames - mean) / std
            encoded.append(model(frames).cpu().numpy().astype(np.float32))

    embeddings = np.concatenate(encoded, axis=0)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        embeddings=embeddings.astype(np.float16),
        timestamps=subject.timestamps,
        labels=subject.labels,
    )
    weights_path = _weights_path()
    return {
        "subject_id": subject_id,
        "status": "EXTRACTED",
        "rgb_cache_path": str(rgb_cache_path),
        "rgb_cache_sha256": sha256_file(rgb_cache_path),
        "output_path": str(output_path),
        "output_sha256": sha256_file(output_path),
        "backbone": "torchvision_resnet18_imagenet1k_v1",
        "weights_path": "torch_hub/checkpoints/resnet18-f37072fd.pth",
        "weights_sha256": sha256_file(weights_path),
        "records": len(embeddings),
        "embedding_dim": int(embeddings.shape[1]),
        "extraction_seconds": round(time.perf_counter() - started, 3),
    }


class EmbeddingClipDataset(Dataset):
    def __init__(
        self,
        subjects: Iterable[SubjectEmbeddings],
        *,
        sequence_length: int,
        stride: int,
        labeled_only: bool,
    ) -> None:
        self.subjects = list(subjects)
        self.items: list[tuple[int, np.ndarray, int]] = []
        center_offset = sequence_length // 2
        for subject_index, subject in enumerate(self.subjects):
            for indices in clip_indices(len(subject.embeddings), sequence_length, stride):
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
        sequence = torch.from_numpy(subject.embeddings[indices].copy()).float()
        return sequence, int(subject.labels[center]), subject_index, center

    def sample_weights(self) -> torch.Tensor:
        labels = np.asarray(
            [int(self.subjects[s].labels[center]) for s, _, center in self.items],
            dtype=np.int64,
        )
        counts = np.bincount(labels, minlength=len(ACTION_CLASSES)).astype(np.float64)
        class_weight = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
        return torch.tensor(class_weight[labels], dtype=torch.double)

    def loss_weights(self) -> torch.Tensor:
        labels = np.asarray(
            [int(self.subjects[s].labels[center]) for s, _, center in self.items],
            dtype=np.int64,
        )
        counts = np.bincount(labels, minlength=len(ACTION_CLASSES)).astype(np.float64)
        weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
        weights /= weights.mean()
        return torch.tensor(weights, dtype=torch.float32)


class ResNetGRUClassifier(nn.Module):
    def __init__(
        self,
        *,
        embedding_dim: int = 512,
        hidden_size: int = 64,
        num_classes: int = len(ACTION_CLASSES),
    ) -> None:
        super().__init__()
        self.input_norm = nn.LayerNorm(embedding_dim)
        self.gru = nn.GRU(
            embedding_dim,
            hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        pooled_dim = hidden_size * 2 * 3
        self.classifier = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.25),
            nn.Linear(128, num_classes),
        )

    def forward(self, sequences: torch.Tensor) -> torch.Tensor:
        encoded, _ = self.gru(self.input_norm(sequences))
        pooled = torch.cat(
            (
                encoded.mean(dim=1),
                encoded.max(dim=1).values,
                encoded[:, -1] - encoded[:, 0],
            ),
            dim=1,
        )
        return self.classifier(pooled)


def train_resnet_gru(
    dataset: EmbeddingClipDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    sampling_mode: str = "sqrt_inverse",
    loss_mode: str = "sqrt_inverse",
) -> tuple[ResNetGRUClassifier, dict[str, Any]]:
    set_deterministic(seed)
    model = ResNetGRUClassifier()
    generator = torch.Generator().manual_seed(seed)
    if sampling_mode == "sqrt_inverse":
        sampler = WeightedRandomSampler(
            dataset.sample_weights(),
            num_samples=len(dataset),
            replacement=True,
            generator=generator,
        )
        loader = DataLoader(dataset, batch_size=batch_size, sampler=sampler, num_workers=0)
    elif sampling_mode == "none":
        loader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
    else:
        raise ValueError(f"unsupported sampling mode: {sampling_mode}")
    if loss_mode == "sqrt_inverse":
        loss_weights = dataset.loss_weights()
    elif loss_mode == "none":
        loss_weights = None
    else:
        raise ValueError(f"unsupported loss mode: {loss_mode}")
    criterion = nn.CrossEntropyLoss(weight=loss_weights, label_smoothing=0.05)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    history: list[dict[str, float]] = []
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
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            correct += int((logits.argmax(dim=1) == labels).sum().item())
            seen += len(labels)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(total_loss / max(seen, 1), 6),
                "sampled_training_accuracy": round(correct / max(seen, 1), 6),
            }
        )
    return model, {
        "epochs": epochs,
        "samples": len(dataset),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_seconds": round(time.perf_counter() - started, 3),
        "sampling": sampling_mode,
        "loss_weighting": loss_mode,
        "history": history,
    }


def predict_embeddings(
    model: ResNetGRUClassifier,
    subject: SubjectEmbeddings,
    *,
    sequence_length: int,
    batch_size: int,
) -> dict[str, Any]:
    dataset = EmbeddingClipDataset(
        [subject], sequence_length=sequence_length, stride=1, labeled_only=False
    )
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    process = psutil.Process(os.getpid())
    rss_before_mb = process.memory_info().rss / (1024 * 1024)
    peak_rss_mb = [rss_before_mb]
    stop_memory_sampling = threading.Event()

    def sample_memory() -> None:
        while not stop_memory_sampling.wait(0.01):
            peak_rss_mb[0] = max(
                peak_rss_mb[0], process.memory_info().rss / (1024 * 1024)
            )

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    probabilities: list[np.ndarray] = []
    centers: list[int] = []
    per_clip_latency_ms: list[float] = []
    started = time.perf_counter()
    model.eval()
    try:
        with torch.inference_mode():
            for sequences, _, _, batch_centers in loader:
                batch_started = time.perf_counter()
                probabilities.append(torch.softmax(model(sequences), dim=1).cpu().numpy())
                elapsed = time.perf_counter() - batch_started
                per_clip_latency_ms.extend(
                    [1000.0 * elapsed / max(len(sequences), 1)] * len(sequences)
                )
                centers.extend(int(value) for value in batch_centers.numpy())
    finally:
        stop_memory_sampling.set()
        sampler.join()
        rss_after_mb = process.memory_info().rss / (1024 * 1024)
        peak_rss_mb[0] = max(peak_rss_mb[0], rss_after_mb)

    elapsed = time.perf_counter() - started
    probs = np.concatenate(probabilities, axis=0)
    centers_array = np.asarray(centers, dtype=np.int64)
    return {
        "subject_id": subject.subject_id,
        "centers": centers_array,
        "timestamps": subject.timestamps[centers_array],
        "labels": subject.labels[centers_array],
        "probabilities": probs,
        "inference_seconds": round(elapsed, 6),
        "clips_per_second": round(len(dataset) / elapsed, 3),
        "milliseconds_per_clip": round(1000.0 * elapsed / max(len(dataset), 1), 3),
        "p50_latency_ms": round(float(np.percentile(per_clip_latency_ms, 50)), 3),
        "p95_latency_ms": round(float(np.percentile(per_clip_latency_ms, 95)), 3),
        "rss_before_mb": round(rss_before_mb, 3),
        "rss_after_mb": round(rss_after_mb, 3),
        "peak_process_rss_mb": round(peak_rss_mb[0], 3),
    }
