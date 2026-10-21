"""Unit and regression test suite for ROADWATCH V23 improvements."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.ai.auxiliary import PoseEvidence
from training.dmd.action_adapter import ActionInferenceAdapter
from training.dmd.action_checkpoint import (
    CheckpointSafetyError,
    sha256_file,
    validate_checkpoint_manifest,
)
from training.dmd.behavior_roi import Box
from training.dmd.driver_phone_association import (
    AssociationState,
    DriverPhoneAssociation,
    OccupantTrack,
)
from training.dmd.run_verified_v23 import (
    OUTPUT,
    load_calibration,
    load_cohort,
    replay_v23_record,
)

ROOT = Path(__file__).resolve().parents[1]


def test_v23_camera_seat_calibration_integrity():
    """Verify camera seat calibration manifest conforms to physical DMD specifications."""
    cal = load_calibration(OUTPUT)
    assert cal["camera_id"] == "BODY"
    assert cal["vehicle_configuration"]["driver_side"] == "left"
    assert len(cal["seats"]) == 2

    driver = cal["seats"][0]
    passenger = cal["seats"][1]
    assert driver["role"] == "driver"
    assert passenger["role"] == "front_passenger"
    assert driver["normalized_xyxy"] == [0.0, 0.10, 0.65, 0.95]
    assert passenger["normalized_xyxy"] == [0.65, 0.10, 1.0, 0.95]


def test_v23_driver_role_without_pose_is_unknown():
    """Verify that absent pose fails closed to UNKNOWN role and 0 confidence."""
    cal = load_calibration(OUTPUT)
    record = {
        "timestamp": 10.0,
        "poses": [],
        "candidates": [
            {
                "box": [100, 100, 200, 200],
                "local_box": [100, 100, 200, 200],
                "confidence": 0.85,
            }
        ],
        "observation": {"timestamp": 10.0},
    }
    obs, detail = replay_v23_record(
        record,
        adapter=None,
        calibration=cal,
        run_spec={"association_mode": "VERIFIED_SEAT_CALIBRATION"},
    )
    assert obs.occupant_role == "UNKNOWN"
    assert obs.confidence == 0.0
    assert obs.fusion_score == 0.0
    assert detail["driver_track_verified"] is False


def test_v23_verified_driver_association_emits_positive_evidence():
    """Verify that pose inside driver seat + phone inside occupant produces DRIVER_ASSOCIATED."""
    cal = load_calibration(OUTPUT)
    # Pose covering driver crop area [100, 50, 700, 550]
    record = {
        "timestamp": 20.0,
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
                "box": [320.0, 220.0, 380.0, 300.0],
                "local_box": [320.0, 220.0, 380.0, 300.0],
                "confidence": 0.75,
            }
        ],
        "observation": {"timestamp": 20.0},
    }
    obs, detail = replay_v23_record(
        record,
        adapter=None,
        calibration=cal,
        run_spec={"association_mode": "VERIFIED_SEAT_CALIBRATION"},
    )
    assert obs.occupant_role == "driver"
    assert obs.confidence == 0.75
    assert obs.evidence_source == "V23_VERIFIED_ASSOCIATION"
    assert detail["driver_track_verified"] is True
    assert detail["rejection_reason"] == "NONE_ACCEPTED"


def test_v23_passenger_phone_does_not_impersonate_driver():
    """Verify that phone belonging to passenger is excluded from driver violations."""
    assoc = DriverPhoneAssociation(minimum_containment=0.35)
    driver = OccupantTrack("d0", "driver", Box(0, 0, 400, 500), True, 0.90)
    passenger = OccupantTrack("p0", "front_passenger", Box(500, 0, 900, 500), True, 0.90)

    passenger_phone = Box(600, 100, 700, 250)
    res = assoc.associate(
        passenger_phone,
        (driver, passenger),
        phone_context="HANDHELD_USE",
        phone_confidence=0.80,
        context_verified=True,
    )
    assert res.state == AssociationState.PASSENGER_ASSOCIATED.value
    assert res.occupant_track_id == "p0"


def test_v23_declared_latency_contract_validation():
    """Verify that DECLARED_LATENCY_CENTERED accepts CENTER models with declared latency."""
    # Strict causal online must fail closed
    with pytest.raises(CheckpointSafetyError, match="CENTERED_TRAINING_NOT_VALIDATED"):
        ActionInferenceAdapter(
            mode="V6_HIERARCHICAL",
            heldout_subject="gC-14",
            causal_mode="STRICT_CAUSAL_ONLINE",
            missing_checkpoint_policy="ERROR",
        )

    # Declared latency centered must succeed
    adapter = ActionInferenceAdapter(
        mode="V6_HIERARCHICAL",
        heldout_subject="gC-14",
        causal_mode="DECLARED_LATENCY_CENTERED",
        missing_checkpoint_policy="ERROR",
    )
    assert adapter.actual_mode == "V6_HIERARCHICAL"
    assert adapter.declared_latency_seconds == 2.0
    prov = adapter.provenance()
    assert prov["causal_mode"] == "DECLARED_LATENCY_CENTERED"
    assert prov["model_loaded"] is True
    assert prov["forward_count"] == 0


def test_v23_deep_model_forward_increments_counter():
    """Verify that feeding 9 frames triggers deep model forward and increments forward_count."""
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
        seq = adapter.observe_embedding(dummy_emb, key, timestamp=i * 0.5)
        assert seq is None
    assert adapter.forward_count == 0

    # Feed 9th frame -> sequence complete
    seq9 = adapter.observe_embedding(dummy_emb, key, timestamp=4.0)
    assert seq9 is not None
    assert seq9.shape == (9, 512)

    # Predict with deep model
    scores = adapter.predict(
        driver_crop=np.zeros((100, 100, 3), dtype=np.uint8),
        pose=PoseEvidence(0.9, ((10.0, 10.0),), ((20.0, 20.0),)),
        phone_conf=0.7,
        phone_box_local=[10, 10, 40, 40],
        session_id="s2",
        camera_id="BODY",
        occupant_track_id="driver_0",
        timestamp=4.5,
    )
    assert adapter.forward_count >= 1
    assert scores.phone_use >= 0.0
    assert scores.safe_driving >= 0.0
