"""Hierarchical temporal action head for governed Roadwatch V6 research.

Stage one separates SAFE_DRIVING, PHONE_USE, and OTHER_ACTIVITY.  Stage two
classifies phone behavior only when the coarse head assigns phone probability.
The hierarchy never turns an action prediction into a verified violation.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.dmd.v5_resnet_gru import EmbeddingClipDataset
from training.dmd.v5_video_action import ACTION_CLASSES, set_deterministic

COARSE_CLASSES = ("SAFE_DRIVING", "PHONE_USE", "OTHER_ACTIVITY")
PHONE_CLASS_INDICES = (1, 2, 3, 4)


def coarse_targets(labels: torch.Tensor) -> torch.Tensor:
    """Map the six flat labels to SAFE, PHONE_USE, or OTHER."""

    targets = torch.full_like(labels, 2)
    targets[labels == 0] = 0
    phone_mask = (labels >= PHONE_CLASS_INDICES[0]) & (labels <= PHONE_CLASS_INDICES[-1])
    targets[phone_mask] = 1
    return targets


def _sqrt_inverse_weights(labels: np.ndarray, classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=classes).astype(np.float64)
    weights = np.sqrt(counts.sum() / np.maximum(counts, 1.0))
    weights /= weights.mean()
    return torch.tensor(weights, dtype=torch.float32)


class HierarchicalResNetGRUClassifier(nn.Module):
    """Shared BiGRU encoder with independent coarse and phone-action heads."""

    def __init__(self, *, embedding_dim: int = 512, hidden_size: int = 64) -> None:
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
        self.shared = nn.Sequential(
            nn.LayerNorm(pooled_dim),
            nn.Linear(pooled_dim, 128),
            nn.SiLU(),
            nn.Dropout(0.25),
        )
        self.coarse_head = nn.Linear(128, len(COARSE_CLASSES))
        self.phone_head = nn.Linear(128, len(PHONE_CLASS_INDICES))

    def forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.gru(self.input_norm(sequences))
        pooled = torch.cat(
            (
                encoded.mean(dim=1),
                encoded.max(dim=1).values,
                encoded[:, -1] - encoded[:, 0],
            ),
            dim=1,
        )
        shared = self.shared(pooled)
        return self.coarse_head(shared), self.phone_head(shared)


def hierarchical_probabilities(
    coarse_logits: torch.Tensor, phone_logits: torch.Tensor
) -> torch.Tensor:
    """Return normalized probabilities in the original six-class order."""

    coarse = torch.softmax(coarse_logits, dim=1)
    phone = torch.softmax(phone_logits, dim=1)
    probabilities = torch.zeros(
        (len(coarse), len(ACTION_CLASSES)),
        dtype=coarse.dtype,
        device=coarse.device,
    )
    probabilities[:, 0] = coarse[:, 0]
    probabilities[:, 5] = coarse[:, 2]
    probabilities[:, 1:5] = coarse[:, 1:2] * phone
    return probabilities


def train_hierarchical_resnet_gru(
    dataset: EmbeddingClipDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    loss_mode: str = "sqrt_inverse",
) -> tuple[HierarchicalResNetGRUClassifier, dict[str, Any]]:
    """Train without replacement sampling so observed class priors remain visible."""

    set_deterministic(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        generator=generator,
        num_workers=0,
    )
    flat_labels = np.asarray(
        [int(dataset.subjects[s].labels[center]) for s, _, center in dataset.items],
        dtype=np.int64,
    )
    coarse_labels = np.where(flat_labels == 0, 0, np.where(flat_labels == 5, 2, 1))
    phone_labels = flat_labels[np.isin(flat_labels, PHONE_CLASS_INDICES)] - 1
    if loss_mode == "sqrt_inverse":
        coarse_weights = _sqrt_inverse_weights(coarse_labels, len(COARSE_CLASSES))
        phone_weights = _sqrt_inverse_weights(phone_labels, len(PHONE_CLASS_INDICES))
    elif loss_mode == "none":
        coarse_weights = None
        phone_weights = None
    else:
        raise ValueError(f"unsupported loss mode: {loss_mode}")
    coarse_criterion = nn.CrossEntropyLoss(weight=coarse_weights, label_smoothing=0.03)
    phone_criterion = nn.CrossEntropyLoss(weight=phone_weights, label_smoothing=0.03)
    model = HierarchicalResNetGRUClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        coarse_correct = 0
        seen = 0
        for sequences, labels, _, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            coarse_logits, phone_logits = model(sequences)
            coarse = coarse_targets(labels)
            loss = coarse_criterion(coarse_logits, coarse)
            phone_mask = (labels >= 1) & (labels <= 4)
            if bool(phone_mask.any()):
                loss = loss + phone_criterion(phone_logits[phone_mask], labels[phone_mask] - 1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(labels)
            coarse_correct += int((coarse_logits.argmax(dim=1) == coarse).sum().item())
            seen += len(labels)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(total_loss / max(seen, 1), 6),
                "sampled_coarse_accuracy": round(coarse_correct / max(seen, 1), 6),
            }
        )
    return model, {
        "epochs": epochs,
        "samples": len(dataset),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_seconds": round(time.perf_counter() - started, 3),
        "sampling": "none",
        "loss_weighting": loss_mode,
        "history": history,
    }


def predict_hierarchical(
    model: HierarchicalResNetGRUClassifier,
    dataset: EmbeddingClipDataset,
    *,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities: list[np.ndarray] = []
    centers: list[int] = []
    coarse_confidence: list[np.ndarray] = []
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for sequences, _, _, batch_centers in loader:
            coarse_logits, phone_logits = model(sequences)
            probabilities.append(
                hierarchical_probabilities(coarse_logits, phone_logits).cpu().numpy()
            )
            coarse_confidence.append(torch.softmax(coarse_logits, dim=1).cpu().numpy())
            centers.extend(int(value) for value in batch_centers.numpy())
    elapsed = time.perf_counter() - started
    subject = dataset.subjects[0]
    centers_array = np.asarray(centers, dtype=np.int64)
    return {
        "subject_id": subject.subject_id,
        "centers": centers_array,
        "timestamps": subject.timestamps[centers_array],
        "labels": subject.labels[centers_array],
        "probabilities": np.concatenate(probabilities, axis=0),
        "coarse_probabilities": np.concatenate(coarse_confidence, axis=0),
        "inference_seconds": round(elapsed, 6),
        "clips_per_second": round(len(dataset) / max(elapsed, 1e-9), 3),
    }
