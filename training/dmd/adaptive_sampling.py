"""Causal visual sampling and timestamp-based dense inference policies."""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class SamplingConfig:
    base_stride: int = 15
    dense_stride: int = 3
    motion_threshold: float = 0.035
    burst_seconds: float = 0.4


class VisualSampler:
    """Select centers without labels, annotations, or action-model predictions."""

    def __init__(self, config: SamplingConfig | None = None) -> None:
        config = config or SamplingConfig()
        if config.base_stride <= 0 or config.dense_stride <= 0:
            raise ValueError("strides must be positive")
        if not 0 < config.motion_threshold <= 1 or config.burst_seconds < 0:
            raise ValueError("invalid motion configuration")
        self.config = config
        self.previous: np.ndarray | None = None
        self.last_frame = -1
        self.burst_until = -1.0

    def observe(self, frame_index: int, timestamp: float, roi: np.ndarray) -> dict:
        if frame_index <= self.last_frame:
            raise ValueError("frames must be strictly increasing")
        if not np.isfinite(timestamp) or timestamp < 0 or roi.size == 0:
            raise ValueError("invalid timestamp or empty ROI")
        self.last_frame = frame_index
        gray = cv2.resize(cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY), (64, 64))
        gray = gray.astype(np.float32) / 255.0
        motion = (
            float(np.abs(gray - self.previous).mean())
            if self.previous is not None
            else 0.0
        )
        self.previous = gray
        triggered = motion >= self.config.motion_threshold
        if triggered:
            self.burst_until = timestamp + self.config.burst_seconds
        baseline = frame_index % self.config.base_stride == 0
        dense = frame_index % self.config.dense_stride == 0
        adaptive = baseline or (dense and triggered)
        burst = baseline or (dense and timestamp <= self.burst_until)
        return {
            "motion": motion,
            "T0": baseline,
            "T1": dense,
            "T2": adaptive,
            "T3": baseline or frame_index % (2 * self.config.dense_stride) == 0,
            "T4": burst,
            "T5": burst,
        }


def padded_roi(
    shape: tuple, xyxy: tuple[float, ...], padding: float = 0.0
) -> tuple[int, ...]:
    """Clip a normalized ROI to the image; padding is a fraction of ROI extent."""
    h, w = shape[:2]
    x1, y1, x2, y2 = xyxy
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1) or padding < 0:
        raise ValueError("invalid ROI or padding")
    dx, dy = (x2 - x1) * padding, (y2 - y1) * padding
    return (
        max(0, int(np.floor((x1 - dx) * w + 1e-9))),
        max(0, int(np.floor((y1 - dy) * h + 1e-9))),
        min(w, int(np.ceil((x2 + dx) * w - 1e-9))),
        min(h, int(np.ceil((y2 + dy) * h - 1e-9))),
    )


def restore_boxes(boxes: np.ndarray, roi: tuple[int, ...], shape: tuple) -> np.ndarray:
    """Restore source-crop boxes to full-frame pixel coordinates."""
    boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4).copy()
    boxes += np.asarray([roi[0], roi[1], roi[0], roi[1]], dtype=np.float32)
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, shape[1])
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, shape[0])
    return boxes


def confirmed_events(
    timestamps: np.ndarray,
    scores: np.ndarray,
    *,
    threshold: float = 0.8,
    release: float = 0.52,
    maximum_gap: float = 0.6,
) -> list[dict]:
    """Require two distinct positive samples and preserve evidence endpoints."""
    if len(timestamps) != len(scores) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("aligned, strictly increasing timestamps required")
    if not np.isfinite(timestamps).all() or not np.isfinite(scores).all():
        raise ValueError("non-finite evidence")
    events: list[dict] = []
    run: list[tuple[float, float]] = []
    confirmed: float | None = None

    def finish() -> None:
        nonlocal run, confirmed
        if confirmed is not None:
            events.append(
                {
                    "event_type": "PHONE",
                    "start_seconds": float(run[0][0]),
                    "end_seconds": float(run[-1][0]),
                    "duration": float(run[-1][0] - run[0][0]),
                    "confirmed_at_seconds": float(confirmed),
                    "confidence": float(max(value for _, value in run)),
                    "source": "ADAPTIVE_ACTION",
                    "review_status": "NEEDS_REVIEW",
                }
            )
        run, confirmed = [], None

    for timestamp, score in zip(timestamps, scores, strict=True):
        if run and (timestamp - run[-1][0] > maximum_gap or score < release):
            finish()
        if not run:
            if score >= threshold:
                run = [(timestamp, score)]
        elif confirmed is not None or score >= threshold:
            run.append((timestamp, score))
            if confirmed is None and len(run) >= 2:
                confirmed = timestamp
        else:
            run = []
    finish()
    return events


def fuse_available_views(
    scores: dict[str, float | None], weights: dict[str, float]
) -> float | None:
    """Missing views abstain; they never contribute negative evidence."""
    available = [
        (score, weights.get(view, 0.0))
        for view, score in scores.items()
        if score is not None and weights.get(view, 0.0) > 0
    ]
    if not available:
        return None
    if any(not np.isfinite(score) or not 0 <= score <= 1 for score, _ in available):
        raise ValueError("view scores must be finite probabilities")
    return sum(score * weight for score, weight in available) / sum(
        weight for _, weight in available
    )


class AdaptiveFrameSampler:
    """Switch from a base stride to a dense stride after an evidence trigger."""

    def __init__(
        self,
        *,
        base_stride: int,
        dense_stride: int,
        hold_seconds: float,
    ) -> None:
        if base_stride < 1 or dense_stride < 1:
            raise ValueError("sampling strides must be positive")
        if dense_stride > base_stride:
            raise ValueError("dense_stride must not exceed base_stride")
        if hold_seconds <= 0.0:
            raise ValueError("hold_seconds must be positive")
        self.base_stride = int(base_stride)
        self.dense_stride = int(dense_stride)
        self.hold_seconds = float(hold_seconds)
        self._dense_until = float("-inf")

    def observe(self, timestamp: float, *, trigger: bool) -> None:
        if trigger:
            self._dense_until = max(self._dense_until, timestamp + self.hold_seconds)

    def is_dense(self, timestamp: float) -> bool:
        return timestamp <= self._dense_until

    def next_stride(self, timestamp: float, *, trigger: bool) -> int:
        self.observe(timestamp, trigger=trigger)
        return self.dense_stride if self.is_dense(timestamp) else self.base_stride

    def next_frame_index(
        self,
        frame_index: int,
        timestamp: float,
        fps: float,
        *,
        trigger: bool,
    ) -> int:
        """Return the next absolute sample while preserving the base grid.

        Variable relative strides drift away from frames ``0, base_stride, ...``
        after a dense burst.  Scheduling on absolute grids guarantees adaptive
        sampling is a strict superset of the baseline samples.
        """
        if frame_index < 0:
            raise ValueError("frame_index must be non-negative")
        if fps <= 0.0:
            raise ValueError("fps must be positive")
        self.observe(timestamp, trigger=trigger)
        next_base = ((frame_index // self.base_stride) + 1) * self.base_stride
        next_dense = ((frame_index // self.dense_stride) + 1) * self.dense_stride
        next_dense_timestamp = next_dense / fps
        if next_dense_timestamp <= self._dense_until + 1e-9:
            return min(next_base, next_dense)
        return next_base

    def reset(self) -> None:
        self._dense_until = float("-inf")
