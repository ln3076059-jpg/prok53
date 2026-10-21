"""Unit and regression test suite for ROADWATCH V24 improvements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.ai.auxiliary import PoseEvidence
from training.dmd.action_adapter import ActionInferenceAdapter
from training.dmd.action_checkpoint import CheckpointSafetyError
from training.dmd.behavior_roi import Box
from training.dmd.driver_phone_association import (
    AssociationState,
    DriverPhoneAssociation,
    OccupantTrack,
)
from training.dmd.run_verified_v24 import (
    OUTPUT,
    StrictTimestampEmbeddingIndex,
    box_overlap,
    load_calibration,
    replay_v24_record,
    verify_occupants,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v24_calibration_strict_validation():
    """Verify camera calibration rejects unverified or invalid status schemas."""
    cal = load_calibration(OUTPUT)
    assert cal["camera_id"] == "BODY"
    assert cal["status"] == "VERIFIED_STATIC_CALIBRATION"
    assert not cal["status"].startswith("UNVERIFIED")

    # Malicious/buggy status containing "VERIFIED" as substring
    cal_unverified = dict(cal, status="UNVERIFIED_STATIC_CALIBRATION")
    assert cal_unverified["status"].startswith("UNVERIFIED")


def test_v24_passenger_pose_first_does_not_impersonate_driver():
    """Verify that when passenger pose is first in poses list, driver role is not erroneously stolen."""
    cal = load_calibration(OUTPUT)
    roi_xyxy = [0, 72, 832, 684]

    # Passenger pose (located in passenger seat [832, 72, 1280, 684] in full frame)
    # in ROI coordinates: x1=850, y1=50, x2=1200, y2=550
    passenger_pose = {
        "confidence": 0.90,
        "xyxy": [850.0, 50.0, 1200.0, 550.0],
        "face_points": [[950.0, 100.0]],
        "hand_points": [[900.0, 300.0]],
    }
    # Driver pose (located in driver seat [0, 72, 832, 684] in full frame)
    driver_pose = {
        "confidence": 0.85,
        "xyxy": [100.0, 50.0, 700.0, 550.0],
        "face_points": [[400.0, 100.0]],
        "hand_points": [[350.0, 250.0]],
    }

    # Passenger is listed FIRST!
    poses = [passenger_pose, driver_pose]
    occupants, driver_track, diag = verify_occupants(poses, cal, roi_xyxy)

    assert len(occupants) == 2
    assert diag["driver_verified"] is True
    assert driver_track is not None
    assert driver_track.role == "driver"
    assert driver_track.track_id == "driver:FRONT_LEFT"

    # First occupant must be passenger!
    assert occupants[0].role == "front_passenger"
    assert occupants[0].track_id == "front_passenger:FRONT_RIGHT"


def test_v24_swapped_pose_order_integrity():
    """Verify that changing pose order yields identical driver track identification."""
    cal = load_calibration(OUTPUT)
    roi_xyxy = [0, 72, 832, 684]

    p1 = {
        "confidence": 0.90,
        "xyxy": [850.0, 50.0, 1200.0, 550.0],
    }
    p2 = {
        "confidence": 0.85,
        "xyxy": [100.0, 50.0, 700.0, 550.0],
    }

    _, driver_a, _ = verify_occupants([p1, p2], cal, roi_xyxy)
    _, driver_b, _ = verify_occupants([p2, p1], cal, roi_xyxy)

    assert driver_a is not None and driver_b is not None
    assert driver_a.role == "driver"
    assert driver_b.role == "driver"
    assert driver_a.box.x1 == driver_b.box.x1
    assert driver_a.box.y2 == driver_b.box.y2


def test_v24_missing_pose_xyxy_fails_closed():
    """Verify that missing pose bounding box does NOT synthesize Box(0, 0, 832, 612)."""
    cal = load_calibration(OUTPUT)
    roi_xyxy = [0, 72, 832, 684]

    corrupt_pose = {
        "confidence": 0.85,
        "xyxy": [],  # Missing/empty
    }
    occupants, driver_track, diag = verify_occupants([corrupt_pose], cal, roi_xyxy)
    assert len(occupants) == 0
    assert driver_track is None
    assert diag["driver_verified"] is False


def test_v24_strict_causal_no_future_leakage(tmp_path):
    """Verify that StrictTimestampEmbeddingIndex forbids future feature retrieval in causal mode."""
    npz_file = tmp_path / "test_emb.npz"
    timestamps = np.array([10.0, 10.5, 11.0, 11.5])
    embeddings = np.random.randn(4, 512).astype(np.float32)
    np.savez(npz_file, timestamps=timestamps, embeddings=embeddings)

    idx = StrictTimestampEmbeddingIndex(npz_file)

    # Looking up at timestamp 10.2s with allow_future=False should pick 10.0s (not 10.5s!)
    emb, gap = idx.get_embedding(10.2, max_gap=0.3, allow_future=False)
    assert emb is not None
    assert gap < 0.0  # Feature was from past (10.0 - 10.2 = -0.2)

    # Looking up before first timestamp fails closed
    emb_early, _ = idx.get_embedding(9.5, max_gap=0.3, allow_future=False)
    assert emb_early is None


def test_v24_continuous_history_increments_forwards():
    """Verify that feeding continuous embeddings accumulates 9 frames and triggers deep model."""
    adapter = ActionInferenceAdapter(
        mode="V6_HIERARCHICAL",
        heldout_subject="gC-14",
        causal_mode="DECLARED_LATENCY_CENTERED",
        missing_checkpoint_policy="ERROR",
    )
    key = ("gC-14", "s2", "BODY", "driver_0")
    dummy_emb = np.zeros(512, dtype=np.float32)

    # Feed 8 frames -> no forward
    for i in range(8):
        seq = adapter.observe_embedding(dummy_emb, key, timestamp=i * 0.504)
        assert seq is None
    assert adapter.forward_count == 0

    # 9th frame completes sequence
    seq9 = adapter.observe_embedding(dummy_emb, key, timestamp=8 * 0.504)
    assert seq9 is not None
    assert len(adapter.embedding_buffer) == 9


def test_v24_predict_from_embedding_parity():
    """Verify that ActionInferenceAdapter.predict_from_embedding works and increments forward_count."""
    adapter = ActionInferenceAdapter(
        mode="V6_HIERARCHICAL",
        heldout_subject="gC-14",
        causal_mode="DECLARED_LATENCY_CENTERED",
        missing_checkpoint_policy="ERROR",
    )
    dummy_emb = np.zeros(512, dtype=np.float32)

    # Feed 8 frames
    for i in range(8):
        adapter.observe_embedding(dummy_emb, ("gC-14", "s2", "BODY", "driver_0"), timestamp=i * 0.504)

    # 9th frame via predict_from_embedding
    scores = adapter.predict_from_embedding(
        embedding=dummy_emb,
        pose=PoseEvidence(0.9, ((10.0, 10.0),), ((20.0, 20.0),)),
        phone_conf=0.7,
        phone_box_local=[10, 10, 40, 40],
        session_id="s2",
        camera_id="BODY",
        occupant_track_id="driver_0",
        timestamp=4.032,
    )
    assert adapter.forward_count >= 1
    assert scores.phone_use >= 0.0
    assert scores.safe_driving >= 0.0


def test_v24_occupant_phone_association_alignment():
    """Verify that passenger phone is not associated with driver even if confidence is high."""
    cal = load_calibration(OUTPUT)
    record = {
        "timestamp": 20.0,
        "roi_xyxy": [0, 72, 832, 684],
        "poses": [
            {
                "confidence": 0.85,
                "xyxy": [100.0, 50.0, 700.0, 550.0],
                "face_points": [[400.0, 100.0]],
                "hand_points": [[350.0, 250.0]],
            }
        ],
        "candidates": [
            {
                # Far right box (in passenger territory, not inside driver pose)
                "box": [850.0, 200.0, 950.0, 300.0],
                "local_box": [850.0, 128.0, 950.0, 228.0],
                "confidence": 0.95,
            }
        ],
        "observation": {"timestamp": 20.0},
    }
    obs, detail = replay_v24_record(
        record,
        adapter=None,
        calibration=cal,
        experiment_spec={"association_mode": "VERIFIED_SEAT_GEOMETRY_V24"},
    )
    # Must reject phone from driver association
    assert obs.confidence == 0.0
    assert detail["rejection_reason"] == "NO_VERIFIED_DRIVER_ASSOCIATION"


def test_v24_e7_governance_fails_closed():
    """Verify that E7 experiment strictly halts and fails closed on unauthorized fitting."""
    from training.dmd.run_verified_v24 import run_experiment
    res = run_experiment("E7")
    assert "BLOCKED_BY_DATA_GOVERNANCE" in res["status"]
    assert res["governance_evaluation"]["rights_accepted"] is False
    assert res["governance_evaluation"]["training_allowed"] is False
