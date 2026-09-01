from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class SequenceEvidence:
    ratio: float
    duration_seconds: float
    observations: int
    smoothed_score: float
    positive_ratio: float = 0.0
    maximum_gap_seconds: float = 0.0


class TemporalFeatureBuffer:
    """Online sequence features used by fusion before the final event persistence gate."""

    def __init__(
        self, window_seconds: float = 2.5, alpha: float = 0.35, positive_score: float = 0.5
    ):
        self.window_seconds = float(window_seconds)
        self.alpha = float(alpha)
        self.positive_score = float(positive_score)
        self.windows: defaultdict[tuple, deque[tuple[float, float]]] = defaultdict(deque)
        self.smoothed: dict[tuple, float] = {}

    def add(self, key: tuple, timestamp: float, score: float) -> SequenceEvidence:
        window = self.windows[key]
        window.append((float(timestamp), float(score)))
        while window and timestamp - window[0][0] > self.window_seconds:
            window.popleft()
        previous = self.smoothed.get(key, float(score))
        smoothed = self.alpha * float(score) + (1 - self.alpha) * previous
        self.smoothed[key] = smoothed
        positives = sum(value >= self.positive_score for _, value in window)
        duration = window[-1][0] - window[0][0] if len(window) > 1 else 0.0
        gaps = [right[0] - left[0] for left, right in zip(window, list(window)[1:], strict=False)]
        return SequenceEvidence(
            ratio=positives / len(window),
            duration_seconds=duration,
            observations=len(window),
            smoothed_score=smoothed,
            positive_ratio=positives / len(window),
            maximum_gap_seconds=max(gaps, default=0.0),
        )

    def reset(self, key: tuple) -> None:
        self.windows.pop(key, None)
        self.smoothed.pop(key, None)

    def reset_prefix(self, prefix: tuple) -> None:
        for key in [key for key in self.windows if key[: len(prefix)] == prefix]:
            self.reset(key)
