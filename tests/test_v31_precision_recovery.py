"""Tests for Roadwatch V3.1 Precision Recovery, Gating, and Deduplication."""
from __future__ import annotations

import pytest
import numpy as np
from pathlib import Path

from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.action_model import RuleGuidedPoseClassifier
from training.dmd.fusion import MultiViewFusionEngine
from training.dmd.holdout_guard import (
    HoldoutAccessError,
    assert_not_consumed_holdout_for_development,
    is_consumed_holdout,
)
from training.dmd.pose_features import PoseFeatureVector
from training.dmd.v3_evaluator import deduplicate_and_merge_events


def _make_dummy_pose(near_ear: bool = False, in_lap: bool = False) -> PoseFeatureVector:
    """Create a synthetic 16D pose feature vector."""
    ear_dist = 0.25 if near_ear else 0.85
    lap_dist = 0.30 if in_lap else 0.90
    return PoseFeatureVector(
        left_wrist_to_left_ear=ear_dist,
        left_wrist_to_right_ear=0.85,
        right_wrist_to_right_ear=0.85,
        right_wrist_to_left_ear=0.85,
        left_wrist_to_nose=ear_dist + 0.05,
        right_wrist_to_nose=0.85,
        left_wrist_to_left_shoulder=0.30 if near_ear else 0.50,
        right_wrist_to_right_shoulder=0.50,
        left_elbow_angle_deg=55.0 if near_ear else 110.0,
        right_elbow_angle_deg=110.0,
        phone_to_left_wrist=lap_dist,
        phone_to_right_wrist=lap_dist,
        phone_to_face=0.30 if near_ear else 0.95,
        visible_keypoint_count=15.0,
        pose_confidence=0.90,
        scale_normalizer=250.0,
    )


def test_cross_view_deduplication():
    candidates = [
        {"event_type": "PHONE", "start_seconds": 10.0, "end_seconds": 15.0, "confidence": 0.8},
        {"event_type": "PHONE", "start_seconds": 16.5, "end_seconds": 22.0, "confidence": 0.85}, # gap 1.5s <= 4.0s
        {"event_type": "PHONE", "start_seconds": 35.0, "end_seconds": 40.0, "confidence": 0.75}, # gap 13s > 4.0s
    ]
    merged = deduplicate_and_merge_events(candidates, merge_gap_seconds=4.0)
    assert len(merged) == 2
    assert merged[0]["start_seconds"] == 10.0
    assert merged[0]["end_seconds"] == 22.0
    assert merged[0]["duration"] == 12.0
    assert merged[1]["start_seconds"] == 35.0


def test_auxiliary_rescue_gating_rejects_spurious_face():
    engine = MultiViewFusionEngine(
        phone_conf_threshold=0.25,
        auxiliary_rescue_mode=True,
        aux_rescue_min_conf=0.35,
    )
    # Driver hands on steering wheel (not near ear)
    pose = _make_dummy_pose(near_ear=False, in_lap=False)
    obs = engine.fuse_frame(
        timestamp_seconds=1.0,
        frame_index=1,
        body_phone_conf=0.0,
        face_phone_conf=0.36, # moderate face detector noise
        hands_phone_conf=0.0,
        body_pose_features=pose,
        body_available=True,
        face_available=True,
        hands_available=True,
    )
    # Spurious face detection must be rejected by rescue gate
    assert not obs.phone_detected
    assert obs.fused_score < 0.25


def test_auxiliary_rescue_gating_allows_occluded_ear_call():
    engine = MultiViewFusionEngine(
        phone_conf_threshold=0.25,
        auxiliary_rescue_mode=True,
        aux_rescue_min_conf=0.35,
    )
    # Driver has hand at left ear (occluded from center BODY camera)
    pose = _make_dummy_pose(near_ear=True, in_lap=False)
    obs = engine.fuse_frame(
        timestamp_seconds=1.0,
        frame_index=1,
        body_phone_conf=0.0, # occluded in BODY
        face_phone_conf=0.55, # visible in FACE
        hands_phone_conf=0.0,
        body_pose_features=pose,
        body_available=True,
        face_available=True,
        hands_available=True,
    )
    # FACE rescue gate must activate and confirm phone use
    assert obs.phone_detected
    assert obs.fused_score >= 0.25
    assert obs.action_type in ("PHONECALL_LEFT", "PHONECALL_RIGHT")


def test_auxiliary_rescue_gating_rejects_spurious_hands():
    engine = MultiViewFusionEngine(
        phone_conf_threshold=0.25,
        auxiliary_rescue_mode=True,
        aux_rescue_min_conf=0.35,
    )
    # Neutral pose, hands at 10-and-2
    pose = _make_dummy_pose(near_ear=False, in_lap=False)
    obs = engine.fuse_frame(
        timestamp_seconds=1.0,
        frame_index=1,
        body_phone_conf=0.0,
        face_phone_conf=0.0,
        hands_phone_conf=0.38, # reflection/watch
        body_pose_features=pose,
        body_available=True,
        face_available=True,
        hands_available=True,
    )
    # Hands reflection must be rejected by rescue gate
    assert not obs.phone_detected
    assert obs.fused_score < 0.25


def test_auxiliary_rescue_gating_allows_lap_texting():
    engine = MultiViewFusionEngine(
        phone_conf_threshold=0.25,
        auxiliary_rescue_mode=True,
        aux_rescue_min_conf=0.35,
    )
    # Driver hands in lap interacting with phone
    pose = _make_dummy_pose(near_ear=False, in_lap=True)
    obs = engine.fuse_frame(
        timestamp_seconds=1.0,
        frame_index=1,
        body_phone_conf=0.0, # occluded by steering wheel
        face_phone_conf=0.0,
        hands_phone_conf=0.60, # visible from hands close-up
        body_pose_features=pose,
        body_available=True,
        face_available=True,
        hands_available=True,
    )
    assert obs.phone_detected
    assert obs.fused_score >= 0.25
    assert obs.action_type == "TEXTING"


def test_two_view_consensus_elevates_confidence():
    engine = MultiViewFusionEngine(
        phone_conf_threshold=0.25,
        auxiliary_rescue_mode=True,
    )
    pose = _make_dummy_pose(near_ear=False, in_lap=False)
    # Both BODY and FACE detect phone at moderate confidence
    obs = engine.fuse_frame(
        timestamp_seconds=1.0,
        frame_index=1,
        body_phone_conf=0.22,
        face_phone_conf=0.28,
        hands_phone_conf=0.0,
        body_pose_features=pose,
        body_available=True,
        face_available=True,
        hands_available=True,
    )
    assert obs.fused_score >= 0.20


def test_stale_track_expiration():
    engine = TemporalEventEngine(
        window_seconds=2.0,
        min_positive_seconds=0.5,
        min_observations=2,
        candidate_threshold=0.25,
        activation_threshold=0.40,
        release_threshold=0.15,
        occlusion_bridge_seconds=2.0, # V3.1 audited duration
    )
    # Step 1: establish active phone detection
    engine.add(Observation(
        timestamp=10.0,
        class_name="phone",
        confidence=0.60,
        track_id=1,
        occupant_role="driver",
        vehicle_context_id="DMD_CABIN",
        phone_hand_proximity=0.8,
    ))
    engine.add(Observation(
        timestamp=10.5,
        class_name="phone",
        confidence=0.60,
        track_id=1,
        occupant_role="driver",
        vehicle_context_id="DMD_CABIN",
        phone_hand_proximity=0.8,
    ))
    # Step 2: phone is put away at t=11.0s, bridge decay until t=13.0s
    obs_bridge = engine.add(Observation(
        timestamp=11.5,
        class_name="phone",
        confidence=0.0,
        track_id=1,
        occupant_role="driver",
        vehicle_context_id="DMD_CABIN",
        phone_hand_proximity=0.8,
    ))
    # Bridge should be active at t=11.5s (gap 0.5s <= 2.0s)
    phone_key = ("DMD_CABIN", 1, "driver")
    assert phone_key in engine.last_physical_phone

    # Step 3: at t=14.0s (gap 3.0s > 2.0s), ghost track must expire cleanly
    engine.add(Observation(
        timestamp=14.0,
        class_name="phone",
        confidence=0.0,
        track_id=1,
        occupant_role="driver",
        vehicle_context_id="DMD_CABIN",
        phone_hand_proximity=0.8,
    ))
    assert phone_key not in engine.last_physical_phone


def test_missing_face_view_graceful_fallback():
    engine = MultiViewFusionEngine(enable_face_fusion=True, auxiliary_rescue_mode=True)
    obs = engine.fuse_frame(
        timestamp_seconds=1.0,
        frame_index=1,
        body_phone_conf=0.75,
        face_phone_conf=0.0,
        hands_phone_conf=0.0,
        body_available=True,
        face_available=False, # camera disconnect
        hands_available=True,
    )
    assert obs.phone_detected
    assert obs.governance_status == "CONFIRMED"


def test_missing_hands_view_graceful_fallback():
    engine = MultiViewFusionEngine(enable_hands_fusion=True, auxiliary_rescue_mode=True)
    obs = engine.fuse_frame(
        timestamp_seconds=1.0,
        frame_index=1,
        body_phone_conf=0.75,
        face_phone_conf=0.0,
        hands_phone_conf=0.0,
        body_available=True,
        face_available=True,
        hands_available=False, # camera disconnect
    )
    assert obs.phone_detected
    assert obs.governance_status == "CONFIRMED"


def test_weak_single_view_evidence_fails_closed():
    engine = MultiViewFusionEngine(phone_conf_threshold=0.25, auxiliary_rescue_mode=True)
    obs = engine.fuse_frame(
        timestamp_seconds=1.0,
        frame_index=1,
        body_phone_conf=0.08, # weak noise
        face_phone_conf=0.0,
        hands_phone_conf=0.0,
        body_available=True,
    )
    assert not obs.phone_detected
    assert obs.governance_status == "FAIL_CLOSED_NO_PHONE"


def test_passenger_phone_isolation():
    engine = TemporalEventEngine()
    # Passenger holding phone must be filtered out before event generation
    candidates = engine.add(Observation(
        timestamp=1.0,
        class_name="phone",
        confidence=0.90,
        track_id=2,
        occupant_role="front_passenger", # NOT driver
        vehicle_context_id="DMD_CABIN",
    ))
    assert len(candidates) == 0


def test_multiview_conflicting_evidence():
    engine = MultiViewFusionEngine(auxiliary_rescue_mode=True)
    # BODY shows neutral driving with no ear proximity; FACE has weak transient detection
    pose = _make_dummy_pose(near_ear=False, in_lap=False)
    obs = engine.fuse_frame(
        timestamp_seconds=1.0,
        frame_index=1,
        body_phone_conf=0.0,
        face_phone_conf=0.20,
        hands_phone_conf=0.0,
        body_pose_features=pose,
        body_available=True,
        face_available=True,
        hands_available=True,
    )
    # Contradiction: BODY shows neutral hands, auxiliary cue is weak -> fail closed
    assert not obs.phone_detected


def test_consumed_holdout_access_protection_permanent():
    for holdout in ["gZ-37", "gE-28"]:
        assert is_consumed_holdout(holdout)
        for action in ["training", "tuning", "ablation", "calibration", "hard_negative_mining"]:
            with pytest.raises(HoldoutAccessError):
                assert_not_consumed_holdout_for_development(holdout, caller_action=action)
