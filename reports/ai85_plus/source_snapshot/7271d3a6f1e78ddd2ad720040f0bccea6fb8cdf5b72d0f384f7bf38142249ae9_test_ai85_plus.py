from pathlib import Path

import numpy as np
import pytest

from training.data_paths import data_path
from training.dmd.adaptive_sampling import (
    SamplingConfig,
    VisualSampler,
    confirmed_events,
    fuse_available_views,
    padded_roi,
    restore_boxes,
)
from training.dmd.holdout_guard import HoldoutAccessError


def test_root_resolution_requires_explicit_choice_for_ambiguity(tmp_path, monkeypatch):
    monkeypatch.delenv("ROADWATCH_DATA_ROOT", raising=False)
    (tmp_path / "dataset").mkdir()
    assert data_path("schemas/a.json", project_root=tmp_path) == tmp_path / "dataset/schemas/a.json"
    (tmp_path / "datasets").mkdir()
    with pytest.raises(ValueError, match="ROADWATCH_DATA_ROOT"):
        data_path("schemas/a.json", project_root=tmp_path)
    monkeypatch.setenv("ROADWATCH_DATA_ROOT", "datasets")
    assert (
        data_path("schemas/a.json", project_root=tmp_path) == tmp_path / "datasets/schemas/a.json"
    )
    monkeypatch.setenv("ROADWATCH_DATA_ROOT", "missing")
    with pytest.raises(FileNotFoundError):
        data_path("a", project_root=tmp_path)


def test_root_rejects_traversal(tmp_path):
    with pytest.raises(ValueError):
        data_path("../private", project_root=tmp_path)


def test_visual_sampling_is_prefix_causal_and_baseline_preserving():
    frames = [np.full((32, 32, 3), v, np.uint8) for v in [0, 0, 255, 255, 255, 255, 0]]

    def run(inputs):
        sampler = VisualSampler()
        return [sampler.observe(i * 3, i * 0.1, frame) for i, frame in enumerate(inputs)]

    whole = run(frames)
    assert whole[:4] == run(frames[:4])
    assert whole[2]["T2"] and whole[3]["T4"] and not whole[3]["T2"]
    assert all(
        not row["T0"] or all(row[v] for v in ("T1", "T2", "T3", "T4", "T5")) for row in whole
    )


def test_confirmation_never_extends_or_duplicates_evidence():
    times = np.array([0.0, 0.1, 0.2, 0.3, 1.5, 1.6, 1.7])
    events = confirmed_events(times, np.array([0.9, 0.9, 0.6, 0.1, 0.9, 0.9, 0.1]))
    assert [(e["start_seconds"], e["end_seconds"]) for e in events] == [(0.0, 0.2), (1.5, 1.6)]
    assert events[0]["confirmed_at_seconds"] == 0.1
    assert all(e["review_status"] == "NEEDS_REVIEW" for e in events)
    with pytest.raises(ValueError):
        confirmed_events(np.array([0.0, 0.0]), np.array([0.9, 0.9]))
    assert confirmed_events(np.array([0.0]), np.array([0.99])) == []


def test_roi_mapping_padding_and_missing_views():
    shape = (100, 200, 3)
    roi = padded_roi(shape, (0.1, 0.2, 0.5, 0.8), 0.25)
    assert roi == (0, 5, 120, 95)
    np.testing.assert_array_equal(restore_boxes([[10, 10, 30, 40]], roi, shape), [[10, 15, 30, 45]])
    assert fuse_available_views({"body": 0.8, "hands": None}, {"body": 1, "hands": 1}) == 0.8
    assert fuse_available_views({"body": None}, {"body": 1}) is None


def test_extractor_rejects_incompatible_time_grid_before_io(tmp_path, monkeypatch):
    from training.dmd.run_ai85_plus_temporal import extract

    def unexpected(*args, **kwargs):
        raise AssertionError("incompatible sampling caused filesystem access")

    monkeypatch.setattr(Path, "exists", unexpected)
    with pytest.raises(ValueError, match="stride"):
        extract("gB-9", tmp_path, SamplingConfig(base_stride=10))


@pytest.mark.parametrize("subject", ["gZ-37", "gE-28"])
def test_new_extractor_blocks_holdouts_before_any_io(subject, tmp_path, monkeypatch):
    from training.dmd.run_ai85_plus_temporal import extract

    def unexpected(*args, **kwargs):
        raise AssertionError("holdout caused filesystem access")

    monkeypatch.setattr(Path, "exists", unexpected)
    with pytest.raises(HoldoutAccessError):
        extract(subject, tmp_path)
