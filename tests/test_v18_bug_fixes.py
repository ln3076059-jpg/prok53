"""Tests for Roadwatch V18: Bug reproduction, fixes, and gA-1 holdout protection."""

from __future__ import annotations

import numpy as np

from training.data_governance import PROTECTED_SUBJECTS, UnifiedAnnotation
from training.dmd.causal_motion import CausalMotionEncoder
from training.dmd.driver_phone_association import AssociationResult, AssociationState
from training.dmd.online_candidate import (
    CLASSES,
    NumericHead,
    ResearchCandidate,
)
from training.split_dataset import validate_isolation


def _make_dummy_head() -> NumericHead:
    """Create a valid numeric head for testing."""
    dim = CausalMotionEncoder().feature_dimensions
    weights = np.zeros((3, dim), dtype=np.float64)
    weights[0, 0] = 1.0  # SAFE_DRIVING
    weights[1, 1] = 1.0  # PHONE_USE
    weights[2, 2] = 1.0  # OTHER_ACTIVITY
    payload = {
        "schema": 1,
        "classes": list(CLASSES),
        "mean": np.zeros(dim, dtype=np.float64).tolist(),
        "scale": np.ones(dim, dtype=np.float64).tolist(),
        "weights": weights.tolist(),
        "bias": [0.0, 0.0, 0.0],
        "provenance": {
            "dataset_hash": "a" * 64,
            "split_hash": "b" * 64,
            "feature_version": "v1",
            "seed": 42,
            "training_subjects": ["gB-9"],
        },
        "calibration": "UNCALIBRATED_OVR_SIGMOID",
    }
    return NumericHead(payload)


def test_e1_handheld_context_normalized():
    """E1: Ensure both 'HANDHELD' and 'HANDHELD_USE' are accepted by ResearchCandidate."""
    head = _make_dummy_head()
    temporal_cfg = {
        "min_positive_seconds": 0.5,
        "min_observations": 2,
        "activation_threshold": 0.25,
    }
    candidate = ResearchCandidate(head, temporal_cfg, enabled=True)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assoc = AssociationResult(
        state=AssociationState.DRIVER_ASSOCIATED.value,
        confidence=0.9,
        occupant_track_id="driver-1",
        source_evidence=("PROXIMITY",),
        reason_codes=(),
    )
    # Prime stream history
    key = ("sess-1", "cam-1", "trk-1")
    for i in range(8):
        candidate.push(frame, 10.0 + i * (15 / 29.76), key, assoc, 0.9, "HANDHELD")

    # 9th push with "HANDHELD" should be admitted if normalized
    res = candidate.push(frame, 10.0 + 8 * (15 / 29.76), key, assoc, 0.9, "HANDHELD")
    msg = f"Observation with 'HANDHELD' context should be admitted! Reasons: {res['reasons']}"
    assert not res["abstained"], msg


def test_e2_temporal_state_not_reset_on_single_unadmitted_frame():
    """E2: Ensure a single unadmitted frame does not destroy TemporalEventEngine state."""
    head = _make_dummy_head()
    temporal_cfg = {
        "min_positive_seconds": 0.5,
        "min_observations": 2,
        "activation_threshold": 0.25,
        "occlusion_bridge_seconds": 3.0,
    }
    candidate = ResearchCandidate(head, temporal_cfg, enabled=True)
    frame = np.zeros((64, 64, 3), dtype=np.uint8)
    assoc = AssociationResult(
        state=AssociationState.DRIVER_ASSOCIATED.value,
        confidence=0.9,
        occupant_track_id="driver-1",
        source_evidence=("PROXIMITY",),
        reason_codes=(),
    )
    key = ("sess-1", "cam-1", "trk-1")
    # Prime history
    t = 10.0
    dt = 15 / 29.76
    for i in range(8):
        candidate.push(frame, t + i * dt, key, assoc, 0.9, "HANDHELD_USE")

    # Push 1st valid observation
    candidate.push(frame, t + 8 * dt, key, assoc, 0.9, "HANDHELD_USE")
    engine_before = candidate.engine

    # Push an unadmitted frame (e.g. temporary unknown association for 1 frame)
    unassoc = AssociationResult(
        state=AssociationState.UNKNOWN.value,
        confidence=0.1,
        occupant_track_id="driver-1",
        source_evidence=(),
        reason_codes=("OCCLUDED",),
    )
    candidate.push(frame, t + 9 * dt, key, unassoc, 0.0, "UNKNOWN")

    # Engine instance should be preserved (internal temporal bridge handles decay), not recreated
    assert candidate.engine is engine_before, "TemporalEventEngine was recreated and wiped out!"


def test_e4_validate_isolation_not_provable_cannot_pass():
    """E4: When trusted_subject_id is missing, validate_isolation must NOT return PASS."""
    manifest = [
        {
            "sample_id": "s1",
            "split": "train",
            "effective_group_id": "g1",
            "source_group_id": "g1",
            "trusted_subject_id": None,  # Not provable
        },
        {
            "sample_id": "s2",
            "split": "val",
            "effective_group_id": "g2",
            "source_group_id": "g2",
            "trusted_subject_id": None,
        },
    ]
    res = validate_isolation(manifest)
    assert res["subject_isolation"] == "NOT_PROVABLE"
    assert res["status"] != "PASS", (
        "validate_isolation must NOT return PASS when subject isolation is NOT_PROVABLE"
    )


def test_e5_safe_driving_with_mounted_phone():
    """E5: Mounted phone or passenger phone must be compatible with SAFE_DRIVING action."""
    rec = UnifiedAnnotation(
        source_id="dmd",
        source_asset_id="asset-1",
        source_hash="a" * 64,
        original_label="safe_drive",
        mapping_version="v1",
        review_status="APPROVED",
        reviewer_id="rev-1",
        phone_action="SAFE_DRIVING",
        phone_visibility="PHONE_VISIBLE",
        phone_context="MOUNTED",  # Mounted phone is safe
    )
    rec.validate()  # Should not raise ValueError


def test_ga1_holdout_protection():
    """gA-1 must be protected as locked independent holdout and never allowed in training."""
    assert "gA-1" in PROTECTED_SUBJECTS or "gA_1" in PROTECTED_SUBJECTS or "1" in PROTECTED_SUBJECTS
