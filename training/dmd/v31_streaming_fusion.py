"""Append-only causal fusion between live object state and action probabilities.

This module defines the online contract only.  It does not reconstruct object
activation times from finalized event files, because doing that would hide the
very latency that the contract is meant to measure.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from training.dmd.v30_causal_action import (
    CLASS_TO_SOURCE_ACTION,
    SOURCE_ACTION_TO_CLASS,
)


@dataclass(frozen=True)
class StreamingFusionConfig:
    action_threshold: float = 0.60
    label_update_margin: float = 0.10
    probability_ema_alpha: float = 0.35
    allow_action_only_activation: bool = False

    def __post_init__(self) -> None:
        if not 0.0 < self.action_threshold <= 1.0:
            raise ValueError("action_threshold must be in (0, 1]")
        if not 0.0 <= self.label_update_margin <= 1.0:
            raise ValueError("label_update_margin must be in [0, 1]")
        if not 0.0 < self.probability_ema_alpha <= 1.0:
            raise ValueError("probability_ema_alpha must be in (0, 1]")
        if self.allow_action_only_activation:
            raise ValueError(
                "action-only activation is not authorized by the V31 causal contract"
            )


@dataclass(frozen=True)
class FusionDecision:
    sequence: int
    decision_type: str
    decision_time_seconds: float
    event_start_seconds: float
    event_end_seconds: float | None
    source_action: str
    action_confidence: float
    object_confidence: float
    evidence_available_through_seconds: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class StreamingObjectActionFusion:
    """Consume timestamp-ordered live signals and emit immutable decisions."""

    def __init__(self, config: StreamingFusionConfig = StreamingFusionConfig()) -> None:
        self.config = config
        self._last_timestamp: float | None = None
        self._active = False
        self._event_start = 0.0
        self._current_label = "UNKNOWN"
        self._current_label_score = 0.0
        self._ema: np.ndarray | None = None
        self._sequence = 0
        self._decisions: list[FusionDecision] = []

    @staticmethod
    def _validate_probabilities(probabilities: np.ndarray) -> np.ndarray:
        row = np.asarray(probabilities, dtype=np.float64)
        if row.shape != (6,):
            raise ValueError("action probabilities must have shape (6,)")
        if not np.all(np.isfinite(row)) or np.any(row < 0.0):
            raise ValueError("action probabilities must be finite and non-negative")
        total = float(row.sum())
        if total <= 0.0:
            raise ValueError("action probabilities must have positive mass")
        return row / total

    def _action_label(self) -> tuple[str, float]:
        if self._ema is None:
            return "UNKNOWN", 0.0
        phone_scores = self._ema[1:5]
        local = int(np.argmax(phone_scores))
        class_id = local + 1
        score = float(phone_scores[local])
        if score < self.config.action_threshold:
            return "UNKNOWN", score
        return CLASS_TO_SOURCE_ACTION[class_id], score

    def _append(
        self,
        decision_type: str,
        timestamp: float,
        *,
        object_confidence: float,
        event_end: float | None = None,
    ) -> FusionDecision:
        self._sequence += 1
        decision = FusionDecision(
            sequence=self._sequence,
            decision_type=decision_type,
            decision_time_seconds=round(timestamp, 6),
            event_start_seconds=round(self._event_start, 6),
            event_end_seconds=round(event_end, 6) if event_end is not None else None,
            source_action=self._current_label,
            action_confidence=round(self._current_label_score, 6),
            object_confidence=round(float(object_confidence), 6),
            evidence_available_through_seconds=round(timestamp, 6),
        )
        self._decisions.append(decision)
        return decision

    def update(
        self,
        *,
        timestamp: float,
        object_active: bool,
        object_confidence: float,
        action_probabilities: np.ndarray,
        object_event_start_seconds: float | None = None,
    ) -> list[FusionDecision]:
        """Process one observation; returned decisions are newly emitted only."""

        timestamp = float(timestamp)
        if self._last_timestamp is not None and timestamp <= self._last_timestamp:
            raise ValueError("timestamps must be strictly increasing")
        self._last_timestamp = timestamp
        row = self._validate_probabilities(action_probabilities)
        alpha = self.config.probability_ema_alpha
        self._ema = row if self._ema is None else alpha * row + (1.0 - alpha) * self._ema
        proposed_label, proposed_score = self._action_label()
        emitted: list[FusionDecision] = []

        if not self._active and object_active:
            declared_start = (
                timestamp
                if object_event_start_seconds is None
                else float(object_event_start_seconds)
            )
            if declared_start > timestamp:
                raise ValueError("object event start cannot be later than decision time")
            self._active = True
            self._event_start = declared_start
            self._current_label = proposed_label
            self._current_label_score = proposed_score
            emitted.append(
                self._append("START", timestamp, object_confidence=object_confidence)
            )
            return emitted

        if self._active and not object_active:
            emitted.append(
                self._append(
                    "END",
                    timestamp,
                    object_confidence=object_confidence,
                    event_end=timestamp,
                )
            )
            self._active = False
            self._current_label = "UNKNOWN"
            self._current_label_score = 0.0
            self._ema = None
            return emitted

        if self._active and object_active:
            if (
                object_event_start_seconds is not None
                and round(float(object_event_start_seconds), 6)
                != round(self._event_start, 6)
            ):
                declared_start = float(object_event_start_seconds)
                if declared_start > timestamp:
                    raise ValueError(
                        "object event start cannot be later than decision time"
                    )
                emitted.append(
                    self._append(
                        "END",
                        timestamp,
                        object_confidence=object_confidence,
                        event_end=timestamp,
                    )
                )
                self._ema = row
                proposed_label, proposed_score = self._action_label()
                self._event_start = declared_start
                self._current_label = proposed_label
                self._current_label_score = proposed_score
                emitted.append(
                    self._append(
                        "START", timestamp, object_confidence=object_confidence
                    )
                )
                return emitted
            label_changed = proposed_label != self._current_label
            current_class_id = SOURCE_ACTION_TO_CLASS.get(self._current_label)
            current_ema_score = (
                float(self._ema[current_class_id])
                if current_class_id is not None and self._ema is not None
                else 0.0
            )
            sufficiently_stronger = (
                proposed_score >= current_ema_score + self.config.label_update_margin
            )
            current_unknown = self._current_label == "UNKNOWN" and proposed_label != "UNKNOWN"
            if label_changed and (current_unknown or sufficiently_stronger):
                self._current_label = proposed_label
                self._current_label_score = proposed_score
                emitted.append(
                    self._append(
                        "LABEL_UPDATE",
                        timestamp,
                        object_confidence=object_confidence,
                    )
                )
            elif proposed_label == self._current_label:
                self._current_label_score = max(
                    self._current_label_score, proposed_score
                )
        return emitted

    def decisions(self) -> list[dict[str, Any]]:
        return [decision.to_dict() for decision in self._decisions]

