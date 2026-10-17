"""Past-only motion representations for online driver-behavior inference."""

from __future__ import annotations

from dataclasses import asdict, dataclass

import cv2
import numpy as np


@dataclass(frozen=True)
class CausalMotionConfig:
    lags: tuple[int, ...] = (1, 3, 7)
    map_size: tuple[int, int] = (12, 12)
    include_statistics: bool = True

    def __post_init__(self) -> None:
        if not self.lags or any(lag <= 0 for lag in self.lags):
            raise ValueError("causal lags must be positive")
        if tuple(sorted(set(self.lags))) != self.lags:
            raise ValueError("causal lags must be unique and increasing")
        if min(self.map_size) <= 0:
            raise ValueError("map size must be positive")

    def as_dict(self) -> dict:
        return asdict(self)


def _gray(frame: np.ndarray) -> np.ndarray:
    if frame.ndim == 2:
        return frame.astype(np.float32)
    if frame.ndim == 3 and frame.shape[2] == 3:
        return cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY).astype(np.float32)
    raise ValueError("frame must be grayscale or RGB")


class CausalMotionEncoder:
    """Encode frame t using only frame t and frames with indices less than t."""

    def __init__(self, config: CausalMotionConfig | None = None):
        self.config = config or CausalMotionConfig()

    @property
    def feature_dimensions(self) -> int:
        map_values = self.config.map_size[0] * self.config.map_size[1]
        statistics = 4 if self.config.include_statistics else 0
        return len(self.config.lags) * (map_values + statistics)

    def encode_at(self, frames: np.ndarray, index: int) -> np.ndarray:
        if not 0 <= index < len(frames):
            raise IndexError("frame index is outside the sequence")
        current = _gray(frames[index])
        features = []
        for lag in self.config.lags:
            past_index = index - lag
            if past_index < 0:
                residual = np.zeros_like(current)
            else:
                residual = np.abs(current - _gray(frames[past_index]))
            small = cv2.resize(
                residual, self.config.map_size, interpolation=cv2.INTER_AREA
            )
            features.extend((small / 255.0).ravel())
            if self.config.include_statistics:
                features.extend(
                    (
                        float(residual.mean() / 255.0),
                        float(residual.std() / 255.0),
                        float(np.quantile(residual, 0.90) / 255.0),
                        float(np.mean(residual > 20.0)),
                    )
                )
        return np.asarray(features, dtype=np.float32)

    def encode_sequence(self, frames: np.ndarray) -> np.ndarray:
        if len(frames) == 0:
            return np.empty((0, self.feature_dimensions), dtype=np.float32)
        return np.stack([self.encode_at(frames, index) for index in range(len(frames))])


def causal_running_standardize(
    features: np.ndarray,
    *,
    history: int = 120,
    minimum_history: int = 8,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Normalize each row using only earlier rows from the same online stream."""
    if features.ndim != 2:
        raise ValueError("features must be a two-dimensional array")
    if history <= 0 or minimum_history <= 0 or minimum_history > history:
        raise ValueError("history bounds are invalid")
    output = np.zeros_like(features, dtype=np.float32)
    values = features.astype(np.float64, copy=False)
    prefix = np.vstack((np.zeros((1, values.shape[1])), np.cumsum(values, axis=0)))
    prefix_sq = np.vstack(
        (np.zeros((1, values.shape[1])), np.cumsum(values * values, axis=0))
    )
    for index in range(len(values)):
        start = max(0, index - history)
        count = index - start
        if count < minimum_history:
            continue
        total = prefix[index] - prefix[start]
        total_sq = prefix_sq[index] - prefix_sq[start]
        mean = total / count
        variance = np.maximum(total_sq / count - mean * mean, 0.0)
        output[index] = ((values[index] - mean) / (np.sqrt(variance) + epsilon)).astype(
            np.float32
        )
    return output


def spatial_residual_standardize(
    features: np.ndarray,
    *,
    lags: int = 3,
    map_values: int = 144,
    statistics: int = 4,
    epsilon: float = 1e-3,
) -> np.ndarray:
    """Normalize each residual map spatially while preserving its motion statistics."""
    expected = lags * (map_values + statistics)
    if features.ndim != 2 or features.shape[1] != expected:
        raise ValueError(f"expected feature matrix with {expected} columns")
    result = features.astype(np.float32, copy=True)
    chunk = map_values + statistics
    for lag_index in range(lags):
        start = lag_index * chunk
        maps = result[:, start : start + map_values]
        mean = maps.mean(axis=1, keepdims=True)
        std = maps.std(axis=1, keepdims=True)
        result[:, start : start + map_values] = (maps - mean) / (std + epsilon)
    return result
