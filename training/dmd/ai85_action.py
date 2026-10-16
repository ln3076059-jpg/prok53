"""Checkpoint-compatible temporal attention and controlled CPU fine-tuning.

This research branch reuses the existing ResNet18/BiGRU hierarchy. Zero-initialized
attention reproduces mean pooling before optimization, allowing a matched control.
No labels, statistics or gradients from the outer subject enter fine-tuning.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader

from training.dmd.v5_resnet_gru import EmbeddingClipDataset
from training.dmd.v5_video_action import set_deterministic
from training.dmd.v6_action_hierarchy import (
    HierarchicalResNetGRUClassifier,
    _sqrt_inverse_weights,
    coarse_targets,
)


class AttentionResNetGRUClassifier(HierarchicalResNetGRUClassifier):
    """Replace only the mean component of the inherited temporal pooling."""

    def __init__(self, *, embedding_dim: int = 512, hidden_size: int = 64) -> None:
        super().__init__(embedding_dim=embedding_dim, hidden_size=hidden_size)
        self.temporal_attention = nn.Linear(hidden_size * 2, 1, bias=False)
        nn.init.zeros_(self.temporal_attention.weight)

    def attention_weights(self, encoded: torch.Tensor) -> torch.Tensor:
        return torch.softmax(self.temporal_attention(encoded), dim=1)

    def forward(self, sequences: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        encoded, _ = self.gru(self.input_norm(sequences))
        pooled = torch.cat(
            (
                (encoded * self.attention_weights(encoded)).sum(dim=1),
                encoded.max(dim=1).values,
                encoded[:, -1] - encoded[:, 0],
            ),
            dim=1,
        )
        shared = self.shared(pooled)
        return self.coarse_head(shared), self.phone_head(shared)


def load_inherited_state(
    model: HierarchicalResNetGRUClassifier, state: dict[str, torch.Tensor]
) -> None:
    """Allow exactly the new attention parameter; reject incomplete checkpoints."""
    missing, unexpected = model.load_state_dict(state, strict=False)
    allowed = {"temporal_attention.weight"} if isinstance(
        model, AttentionResNetGRUClassifier
    ) else set()
    if set(missing) != allowed or unexpected:
        raise ValueError(
            f"incompatible inherited checkpoint: missing={missing}, extra={unexpected}"
        )


def finetune(
    model: HierarchicalResNetGRUClassifier,
    dataset: EmbeddingClipDataset,
    *,
    epochs: int,
    batch_size: int,
    learning_rate: float,
    seed: int,
) -> dict[str, Any]:
    """Continue the V6 objective; the caller supplies only training subjects."""
    if not len(dataset) or epochs < 1 or batch_size < 1 or learning_rate <= 0:
        raise ValueError("fine-tuning needs labeled samples and positive training parameters")
    labels = np.asarray(
        [int(dataset.subjects[s].labels[center]) for s, _, center in dataset.items],
        dtype=np.int64,
    )
    if np.any((labels < 0) | (labels > 5)):
        raise ValueError("fine-tuning accepts only the six supervised action classes")
    set_deterministic(seed)
    loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        generator=torch.Generator().manual_seed(seed), num_workers=0,
    )
    coarse = np.where(labels == 0, 0, np.where(labels == 5, 2, 1))
    phone = labels[(labels >= 1) & (labels <= 4)] - 1
    coarse_criterion = nn.CrossEntropyLoss(
        weight=_sqrt_inverse_weights(coarse, 3), label_smoothing=0.03
    )
    phone_criterion = nn.CrossEntropyLoss(
        weight=_sqrt_inverse_weights(phone, 4), label_smoothing=0.03
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-3)
    started = time.perf_counter()
    history = []
    for epoch in range(epochs):
        model.train()
        total, correct, seen = 0.0, 0, 0
        for sequences, flat, _, _ in loader:
            optimizer.zero_grad(set_to_none=True)
            coarse_logits, phone_logits = model(sequences)
            target = coarse_targets(flat)
            loss = coarse_criterion(coarse_logits, target)
            mask = (flat >= 1) & (flat <= 4)
            if bool(mask.any()):
                loss = loss + phone_criterion(phone_logits[mask], flat[mask] - 1)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=2.0)
            optimizer.step()
            total += float(loss.item()) * len(flat)
            correct += int((coarse_logits.argmax(dim=1) == target).sum().item())
            seen += len(flat)
        history.append({
            "epoch": epoch + 1, "loss": total / seen,
            "training_coarse_accuracy": correct / seen,
        })
    return {
        "epochs": epochs, "samples": len(dataset), "seed": seed,
        "batch_size": batch_size, "learning_rate": learning_rate,
        "loss_weighting": "sqrt_inverse", "sampling": "none",
        "checkpoint_selection": "FIXED_FINAL_EPOCH_NO_OUTER_LABEL_SELECTION",
        "optimizer": "FRESH_ADAMW_CONTINUATION_NOT_EXACT_OPTIMIZER_RESUME",
        "training_seconds": time.perf_counter() - started, "history": history,
    }


def negative_evidence_fusion(
    object_events: list[dict[str, Any]],
    timestamps: np.ndarray,
    probabilities: np.ndarray,
    *,
    minimum_safe_probability: float,
    minimum_fraction: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Filter object candidates using probabilities only, without extending events.

    Missing action evidence never vetoes a candidate. This is an ablation, not a
    deployed detector policy. The original event bounds and confidence survive.
    """
    if not 0 < minimum_fraction <= 1 or not 0 <= minimum_safe_probability <= 1:
        raise ValueError("invalid negative-evidence thresholds")
    if probabilities.shape != (len(timestamps), 6):
        raise ValueError("expected one six-class probability row per timestamp")
    retained, audit = [], []
    phone = probabilities[:, 1:5].sum(axis=1)
    for event in object_events:
        mask = (timestamps >= event["start_seconds"]) & (timestamps <= event["end_seconds"])
        safe = probabilities[mask, 0]
        strong = (safe >= minimum_safe_probability) & (safe > phone[mask])
        fraction = float(strong.mean()) if len(safe) else None
        veto = fraction is not None and fraction >= minimum_fraction
        audit.append({
            "event": dict(event), "sample_count": int(mask.sum()),
            "strong_safe_fraction": fraction, "vetoed": veto,
        })
        if not veto:
            retained.append(dict(event))
    return retained, audit
