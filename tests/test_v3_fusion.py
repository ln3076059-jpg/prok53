"""Tests for V3 Multi-View Feature Fusion Engine."""
import numpy as np
import pytest
from training.dmd.fusion import MultiViewFusionEngine, FusedPhoneObservation
from training.dmd.pose_features import PoseFeatureVector


def make_dummy_pose(wrist_to_ear: float = 0.2, elbow_angle: float = 45.0) -> PoseFeatureVector:
    return PoseFeatureVector(
        left_wrist_to_left_ear=wrist_to_ear,
        left_wrist_to_right_ear=1.5,
        right_wrist_to_right_ear=1.5,
        right_wrist_to_left_ear=1.5,
        left_wrist_to_nose=0.3,
        right_wrist_to_nose=1.5,
        left_wrist_to_left_shoulder=0.4,
        right_wrist_to_right_shoulder=1.0,
        left_elbow_angle_deg=elbow_angle,
        right_elbow_angle_deg=180.0,
        phone_to_left_wrist=0.1,
        phone_to_right_wrist=2.0,
        phone_to_face=0.2,
        visible_keypoint_count=10.0,
        pose_confidence=0.85,
        scale_normalizer=150.0,
    )


def test_fusion_missing_body_fail_closed():
    """Missing BODY view must trigger NEEDS_REVIEW fail-closed governance."""
    engine = MultiViewFusionEngine()
    obs = engine.fuse_frame(
        timestamp_seconds=1.0,
        frame_index=30,
        body_available=False,
        face_available=True,
        hands_available=True,
    )
    assert obs.governance_status == "NEEDS_REVIEW"
    assert not obs.phone_detected
    assert obs.fused_score == 0.0


def test_fusion_face_view_recovers_occluded_phone():
    """When phone is occluded in BODY (0.0) but visible in FACE (0.85), fusion recovers detection."""
    engine = MultiViewFusionEngine(phone_conf_threshold=0.25)
    pose = make_dummy_pose(wrist_to_ear=0.2, elbow_angle=40.0)

    obs = engine.fuse_frame(
        timestamp_seconds=2.0,
        frame_index=60,
        body_phone_conf=0.0,  # occluded in center camera
        face_phone_conf=0.85,  # clearly visible in face camera
        body_pose_features=pose,
        body_available=True,
        face_available=True,
        hands_available=False,
    )
    assert obs.phone_detected
    assert obs.fused_score >= 0.25
    assert obs.action_type in ("PHONECALL_LEFT", "PHONECALL_RIGHT")
    assert obs.governance_status == "CONFIRMED"


def test_fusion_never_infers_phone_without_visual_evidence():
    """Even if driver scratches ear (call geometry), without phone detection no violation is inferred."""
    engine = MultiViewFusionEngine(phone_conf_threshold=0.25)
    pose = make_dummy_pose(wrist_to_ear=0.1, elbow_angle=30.0)  # hand near ear

    obs = engine.fuse_frame(
        timestamp_seconds=3.0,
        frame_index=90,
        body_phone_conf=0.0,
        face_phone_conf=0.0,
        hands_phone_conf=0.0,
        body_pose_features=pose,
        body_available=True,
        face_available=True,
        hands_available=True,
    )
    assert not obs.phone_detected
    assert obs.governance_status == "FAIL_CLOSED_NO_PHONE"
    assert obs.fused_score < 0.20
