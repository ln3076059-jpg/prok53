"""Tests for V3 Multi-View Video Synchronization."""
import pytest
from pathlib import Path
from training.dmd.multiview import MultiViewSynchronizer, MultiViewFramePackage
from training.dmd.holdout_guard import HoldoutAccessError


def test_holdout_guard_blocks_ge28(tmp_path):
    """Attempting to initialize MultiViewSynchronizer on gE-28 before freeze must raise HoldoutAccessError."""
    with pytest.raises(HoldoutAccessError):
        MultiViewSynchronizer(tmp_path / "gE-28", "gE-28")


def test_missing_views_behavior(tmp_path):
    """When only BODY is present, FACE and HANDS must be reported unavailable gracefully."""
    subj_dir = tmp_path / "gC-14"
    subj_dir.mkdir()

    # Empty dir has no streams
    sync = MultiViewSynchronizer(subj_dir, "gC-14")
    assert not sync.body_available
    assert not sync.face_available
    assert not sync.hands_available

    aligned, msg = sync.verify_alignment()
    assert not aligned
    assert "BODY view is missing" in msg


def test_frame_package_fail_closed():
    """MultiViewFramePackage must expose explicit availability flags."""
    pkg = MultiViewFramePackage(
        subject="gC-14",
        session="s2",
        frame_index=10,
        timestamp_seconds=0.336,
        body_available=True,
        face_available=False,
        hands_available=False,
        source_hashes={"BODY": "abc123hash"}
    )
    summary = pkg.summary()
    assert summary["body_available"] is True
    assert summary["face_available"] is False
    assert summary["hands_available"] is False
    assert summary["source_hashes"]["BODY"] == "abc123hash"
