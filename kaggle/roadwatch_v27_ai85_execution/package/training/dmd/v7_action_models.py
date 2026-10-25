"""Controlled V7 temporal action models and training objectives."""

from __future__ import annotations

import time
from collections.abc import Collection
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.dmd.v5_resnet_gru import EmbeddingClipDataset
from training.dmd.v5_video_action import ACTION_CLASSES, set_deterministic
from training.dmd.v6_action_hierarchy import (
    PHONE_CLASS_INDICES,
    HierarchicalResNetGRUClassifier,
    _sqrt_inverse_weights,
    coarse_targets,
    hierarchical_probabilities,
)


def _labels(dataset: EmbeddingClipDataset) -> np.ndarray:
    return np.asarray(
        [int(dataset.subjects[s].labels[center]) for s, _, center in dataset.items],
        dtype=np.int64,
    )


def _weights(labels: np.ndarray, classes: int, mode: str) -> torch.Tensor | None:
    if mode == "sqrt_inverse":
        return _sqrt_inverse_weights(labels, classes)
    if mode == "none":
        return None
    raise ValueError(f"unsupported loss mode: {mode}")


class BinaryGateResNetGRUClassifier(nn.Module):
    """PHONE/NON_PHONE gate plus conditional phone and SAFE/OTHER heads."""

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
        self.shared = nn.Sequential(
            nn.LayerNorm(hidden_size * 2 * 3),
            nn.Linear(hidden_size * 2 * 3, 128),
            nn.SiLU(),
            nn.Dropout(0.25),
        )
        self.binary_head = nn.Linear(128, 2)
        self.nonphone_head = nn.Linear(128, 2)
        self.phone_head = nn.Linear(128, 4)

    def forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded, _ = self.gru(self.input_norm(sequences))
        pooled = torch.cat(
            (encoded.mean(dim=1), encoded.max(dim=1).values, encoded[:, -1] - encoded[:, 0]),
            dim=1,
        )
        shared = self.shared(pooled)
        return self.binary_head(shared), self.nonphone_head(shared), self.phone_head(shared)


def binary_gate_probabilities(
    binary_logits: torch.Tensor,
    nonphone_logits: torch.Tensor,
    phone_logits: torch.Tensor,
) -> torch.Tensor:
    binary = torch.softmax(binary_logits, dim=1)
    nonphone = torch.softmax(nonphone_logits, dim=1)
    phone = torch.softmax(phone_logits, dim=1)
    probabilities = torch.zeros(
        (len(binary), len(ACTION_CLASSES)), dtype=binary.dtype, device=binary.device
    )
    probabilities[:, 0] = binary[:, 0] * nonphone[:, 0]
    probabilities[:, 5] = binary[:, 0] * nonphone[:, 1]
    probabilities[:, 1:5] = binary[:, 1:2] * phone
    return probabilities


def train_hierarchical_v7(
    dataset: EmbeddingClipDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
    coarse_loss_mode: str,
    fine_loss_mode: str,
    coarse_loss_coefficient: float,
    fine_loss_coefficient: float,
    hard_negative_centers: Collection[tuple[int, int]] = (),
    hard_negative_weight: float = 1.0,
) -> tuple[HierarchicalResNetGRUClassifier, dict[str, Any]]:
    """Train hierarchy with independently controlled coarse and fine objectives."""

    set_deterministic(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0
    )
    labels = _labels(dataset)
    coarse_labels = np.where(labels == 0, 0, np.where(labels == 5, 2, 1))
    phone_labels = labels[np.isin(labels, PHONE_CLASS_INDICES)] - 1
    coarse_criterion = nn.CrossEntropyLoss(
        weight=_weights(coarse_labels, 3, coarse_loss_mode),
        label_smoothing=0.03,
        reduction="none",
    )
    phone_criterion = nn.CrossEntropyLoss(
        weight=_weights(phone_labels, 4, fine_loss_mode), label_smoothing=0.03
    )
    model = HierarchicalResNetGRUClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    hard_negative_centers = set(hard_negative_centers)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for sequences, flat, subject_indices, centers in loader:
            optimizer.zero_grad(set_to_none=True)
            coarse_logits, phone_logits = model(sequences)
            coarse = coarse_targets(flat)
            sample_loss = coarse_criterion(coarse_logits, coarse)
            if hard_negative_centers:
                multipliers = torch.tensor(
                    [
                        hard_negative_weight
                        if (int(subject), int(center)) in hard_negative_centers
                        else 1.0
                        for subject, center in zip(subject_indices, centers, strict=True)
                    ],
                    dtype=sample_loss.dtype,
                )
                sample_loss = sample_loss * multipliers
            loss = coarse_loss_coefficient * sample_loss.mean()
            phone_mask = (flat >= 1) & (flat <= 4)
            if bool(phone_mask.any()):
                loss = loss + fine_loss_coefficient * phone_criterion(
                    phone_logits[phone_mask], flat[phone_mask] - 1
                )
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(flat)
            correct += int((coarse_logits.argmax(dim=1) == coarse).sum().item())
            seen += len(flat)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(total_loss / max(seen, 1), 6),
                "sampled_coarse_accuracy": round(correct / max(seen, 1), 6),
            }
        )
    return model, {
        "epochs": epochs,
        "samples": len(dataset),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_seconds": round(time.perf_counter() - started, 3),
        "sampling": "none",
        "coarse_loss_mode": coarse_loss_mode,
        "fine_loss_mode": fine_loss_mode,
        "coarse_loss_coefficient": coarse_loss_coefficient,
        "fine_loss_coefficient": fine_loss_coefficient,
        "hard_negative_count": len(hard_negative_centers),
        "hard_negative_weight": hard_negative_weight,
        "history": history,
    }


def mine_training_hard_negatives(
    model: HierarchicalResNetGRUClassifier,
    dataset: EmbeddingClipDataset,
    *,
    batch_size: int,
    minimum_phone_probability: float,
) -> set[tuple[int, int]]:
    """Mine labeled SAFE/OTHER phone errors from training data only."""

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    mined: set[tuple[int, int]] = set()
    model.eval()
    with torch.inference_mode():
        for sequences, labels, subject_indices, centers in loader:
            coarse_logits, phone_logits = model(sequences)
            probabilities = hierarchical_probabilities(coarse_logits, phone_logits)
            phone_probability = probabilities[:, 1:5].sum(dim=1)
            negative = (labels == 0) | (labels == 5)
            selected = negative & (phone_probability >= minimum_phone_probability)
            for subject, center in zip(
                subject_indices[selected], centers[selected], strict=True
            ):
                mined.add((int(subject), int(center)))
    return mined


def train_binary_gate_v7(
    dataset: EmbeddingClipDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> tuple[BinaryGateResNetGRUClassifier, dict[str, Any]]:
    set_deterministic(seed)
    generator = torch.Generator().manual_seed(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, generator=generator, num_workers=0
    )
    labels = _labels(dataset)
    phone_mask_np = np.isin(labels, PHONE_CLASS_INDICES)
    binary_labels = phone_mask_np.astype(np.int64)
    phone_labels = labels[phone_mask_np] - 1
    nonphone_labels = (labels[~phone_mask_np] == 5).astype(np.int64)
    binary_criterion = nn.CrossEntropyLoss(
        weight=_weights(binary_labels, 2, "none"), label_smoothing=0.03
    )
    phone_criterion = nn.CrossEntropyLoss(
        weight=_weights(phone_labels, 4, "sqrt_inverse"), label_smoothing=0.03
    )
    nonphone_criterion = nn.CrossEntropyLoss(
        weight=_weights(nonphone_labels, 2, "none"), label_smoothing=0.03
    )
    model = BinaryGateResNetGRUClassifier()
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    history: list[dict[str, float]] = []
    started = time.perf_counter()
    for epoch in range(epochs):
        model.train()
        total_loss = 0.0
        correct = 0
        seen = 0
        for sequences, flat, _, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            binary_logits, nonphone_logits, phone_logits = model(sequences)
            phone_mask = (flat >= 1) & (flat <= 4)
            binary = phone_mask.long()
            loss = binary_criterion(binary_logits, binary)
            if bool(phone_mask.any()):
                loss = loss + phone_criterion(phone_logits[phone_mask], flat[phone_mask] - 1)
            nonphone_mask = ~phone_mask
            if bool(nonphone_mask.any()):
                nonphone = (flat[nonphone_mask] == 5).long()
                loss = loss + nonphone_criterion(nonphone_logits[nonphone_mask], nonphone)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total_loss += float(loss.item()) * len(flat)
            correct += int((binary_logits.argmax(dim=1) == binary).sum().item())
            seen += len(flat)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": round(total_loss / max(seen, 1), 6),
                "sampled_binary_accuracy": round(correct / max(seen, 1), 6),
            }
        )
    return model, {
        "epochs": epochs,
        "samples": len(dataset),
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "training_seconds": round(time.perf_counter() - started, 3),
        "binary_loss_mode": "none",
        "phone_loss_mode": "sqrt_inverse",
        "nonphone_loss_mode": "none",
        "history": history,
    }


def predict_v7(
    model: nn.Module,
    dataset: EmbeddingClipDataset,
    *,
    architecture: str,
    batch_size: int,
) -> dict[str, Any]:
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=0)
    probabilities: list[np.ndarray] = []
    centers: list[int] = []
    model.eval()
    started = time.perf_counter()
    with torch.inference_mode():
        for sequences, _, _, batch_centers in loader:
            outputs = model(sequences)
            if architecture == "hierarchical":
                batch_probabilities = hierarchical_probabilities(*outputs)
            elif architecture == "binary_gate":
                batch_probabilities = binary_gate_probabilities(*outputs)
            else:
                raise ValueError(f"unsupported architecture: {architecture}")
            probabilities.append(batch_probabilities.cpu().numpy())
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
        "inference_seconds": round(elapsed, 6),
        "clips_per_second": round(len(dataset) / max(elapsed, 1e-9), 3),
    }
