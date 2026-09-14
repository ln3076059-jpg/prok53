"""Tests for V3 Normalized Pose Features."""
import numpy as np
import pytest
from training.dmd.pose_features import (
    extract_pose_features,
    PoseFeatureVector,
    FEATURE_NAMES,
    LEFT_SHOULDER,
    RIGHT_SHOULDER,
    LEFT_ELBOW,
    LEFT_WRIST,
    LEFT_EAR,
    NOSE,
)


def test_feature_vector_structure():
    # 17 keypoints at zeros
    kpts = np.zeros((17, 2), dtype=np.float32)
    confs = np.zeros(17, dtype=np.float32)

    vec = extract_pose_features(kpts, confs)
    assert isinstance(vec, PoseFeatureVector)
    arr = vec.to_array()
    assert len(arr) == len(FEATURE_NAMES)
    assert vec.visible_keypoint_count == 0.0


def test_scale_invariance():
    """Doubling pixel coordinates must preserve normalized distances."""
    def make_person(scale_factor: float):
        kpts = np.zeros((17, 2), dtype=np.float32)
        confs = np.ones(17, dtype=np.float32)

        # Shoulders 100px apart * scale
        kpts[LEFT_SHOULDER] = [200.0 * scale_factor, 200.0 * scale_factor]
        kpts[RIGHT_SHOULDER] = [300.0 * scale_factor, 200.0 * scale_factor]

        # Left ear
        kpts[LEFT_EAR] = [210.0 * scale_factor, 150.0 * scale_factor]

        # Left wrist near left ear (calling posture)
        kpts[LEFT_WRIST] = [215.0 * scale_factor, 155.0 * scale_factor]

        return kpts, confs

    kpts1, confs1 = make_person(1.0)
    kpts2, confs2 = make_person(2.0)

    feat1 = extract_pose_features(kpts1, confs1)
    feat2 = extract_pose_features(kpts2, confs2)

    # Normalized distance should be identical
    assert pytest.approx(feat1.left_wrist_to_left_ear, rel=1e-3) == feat2.left_wrist_to_left_ear
    # Scale normalizer should be doubled
    assert pytest.approx(feat2.scale_normalizer, rel=1e-3) == feat1.scale_normalizer * 2.0


def test_elbow_angle_calling_vs_straight():
    kpts = np.zeros((17, 2), dtype=np.float32)
    confs = np.ones(17, dtype=np.float32)

    # Shoulder at (100, 100), Elbow at (100, 200)
    kpts[LEFT_SHOULDER] = [100.0, 100.0]
    kpts[RIGHT_SHOULDER] = [200.0, 100.0]
    kpts[LEFT_ELBOW] = [100.0, 200.0]

    # Acute bend: wrist up at (100, 110) -> ~10 degrees
    kpts[LEFT_WRIST] = [100.0, 110.0]
    calling_feat = extract_pose_features(kpts, confs)
    assert calling_feat.left_elbow_angle_deg < 30.0

    # Straight arm: wrist down at (100, 300) -> 180 degrees
    kpts[LEFT_WRIST] = [100.0, 300.0]
    straight_feat = extract_pose_features(kpts, confs)
    assert pytest.approx(straight_feat.left_elbow_angle_deg, abs=1.0) == 180.0


def test_phone_proximity_features():
    kpts = np.zeros((17, 2), dtype=np.float32)
    confs = np.ones(17, dtype=np.float32)
    kpts[LEFT_SHOULDER] = [100.0, 100.0]
    kpts[RIGHT_SHOULDER] = [200.0, 100.0]
    kpts[LEFT_WRIST] = [100.0, 150.0]
    kpts[LEFT_EAR] = [100.0, 50.0]

    # Phone box centered at left wrist
    phone_box = (90.0, 140.0, 110.0, 160.0)
    feat = extract_pose_features(kpts, confs, phone_box=phone_box)

    assert feat.phone_to_left_wrist < 0.1  # extremely close
    assert feat.phone_to_right_wrist > 0.5  # far from other hand
