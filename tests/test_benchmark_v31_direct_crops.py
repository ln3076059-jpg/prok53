from __future__ import annotations

import numpy as np

from training.dmd.benchmark_v31_direct_crops import (
    _cosine_summary,
    _legacy_upscale,
    _sharpness,
)


def test_legacy_upscale_has_declared_shape() -> None:
    frames = np.zeros((2, 64, 64, 3), dtype=np.uint8)
    assert _legacy_upscale(frames).shape == (2, 112, 112, 3)


def test_cosine_summary_identical_embeddings_is_one() -> None:
    embeddings = np.eye(3, dtype=np.float32)
    summary = _cosine_summary(embeddings, embeddings)
    assert summary == {"mean": 1.0, "median": 1.0, "minimum": 1.0}


def test_sharpness_constant_image_is_zero() -> None:
    frames = np.zeros((2, 32, 32, 3), dtype=np.uint8)
    summary = _sharpness(frames)
    assert summary["mean_laplacian_variance"] == 0.0
