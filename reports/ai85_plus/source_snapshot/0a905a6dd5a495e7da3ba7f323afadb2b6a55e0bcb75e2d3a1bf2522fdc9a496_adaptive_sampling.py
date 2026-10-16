"""Research-only, causal visual sampling and evidence-bounded confirmation.

Sampling sees only the current frame and past motion. The inherited BiGRU has a
separately declared two-second lookahead; it is not a causal action classifier.
"""

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
        motion = float(np.abs(gray - self.previous).mean()) if self.previous is not None else 0.0
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


def padded_roi(shape: tuple, xyxy: tuple[float, ...], padding: float = 0.0) -> tuple[int, ...]:
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
    """Ultralytics returns boxes in source-crop pixels, without letterbox scale."""
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
    """Require two distinct positive samples; never extrapolate event endpoints.

    No centered smoothing, future frames, gap merging, or duration extension.
    Confirmation timestamp is distinct from the first positive observation.
    """
    if len(timestamps) != len(scores) or np.any(np.diff(timestamps) <= 0):
        raise ValueError("aligned, strictly increasing timestamps required")
    if not np.isfinite(timestamps).all() or not np.isfinite(scores).all():
        raise ValueError("non-finite evidence")
    events, run = [], []
    confirmed = None

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
                    "confidence": float(max(v for _, v in run)),
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
    if any(not np.isfinite(s) or not 0 <= s <= 1 for s, _ in available):
        raise ValueError("view scores must be finite probabilities")
    return sum(s * w for s, w in available) / sum(w for _, w in available)
