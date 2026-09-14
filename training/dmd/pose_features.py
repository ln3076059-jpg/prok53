"""Normalized Driver Pose Feature Extraction for Roadwatch V3.

Extracts normalized geometric features from upper-body keypoints to distinguish:
- Phone calling (hand/wrist to ear, hand/wrist to nose, sharp elbow angle)
- Texting (wrists together in front of torso, wrists below chest, phone near hands)
- Normal driving (both hands near steering wheel, away from ears/nose)
- Non-distracted gestures (drinking, hair touching, scratching)

Normalizes all spatial distances by shoulder width (or torso diagonal fallback)
to remain invariant to camera distance, subject morphology, and resolution.
"""
from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Sequence, Tuple, Optional, Union
import numpy as np

# COCO 17 Keypoint Indices
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12

FEATURE_NAMES = [
    "left_wrist_to_left_ear",
    "left_wrist_to_right_ear",
    "right_wrist_to_right_ear",
    "right_wrist_to_left_ear",
    "left_wrist_to_nose",
    "right_wrist_to_nose",
    "left_wrist_to_left_shoulder",
    "right_wrist_to_right_shoulder",
    "left_elbow_angle_deg",
    "right_elbow_angle_deg",
    "phone_to_left_wrist",
    "phone_to_right_wrist",
    "phone_to_face",
    "visible_keypoint_count",
    "pose_confidence",
    "scale_normalizer",
]


@dataclass(frozen=True)
class PoseFeatureVector:
    left_wrist_to_left_ear: float
    left_wrist_to_right_ear: float
    right_wrist_to_right_ear: float
    right_wrist_to_left_ear: float
    left_wrist_to_nose: float
    right_wrist_to_nose: float
    left_wrist_to_left_shoulder: float
    right_wrist_to_right_shoulder: float
    left_elbow_angle_deg: float
    right_elbow_angle_deg: float
    phone_to_left_wrist: float
    phone_to_right_wrist: float
    phone_to_face: float
    visible_keypoint_count: float
    pose_confidence: float
    scale_normalizer: float

    def to_array(self) -> np.ndarray:
        return np.array([
            self.left_wrist_to_left_ear,
            self.left_wrist_to_right_ear,
            self.right_wrist_to_right_ear,
            self.right_wrist_to_left_ear,
            self.left_wrist_to_nose,
            self.right_wrist_to_nose,
            self.left_wrist_to_left_shoulder,
            self.right_wrist_to_right_shoulder,
            self.left_elbow_angle_deg,
            self.right_elbow_angle_deg,
            self.phone_to_left_wrist,
            self.phone_to_right_wrist,
            self.phone_to_face,
            self.visible_keypoint_count,
            self.pose_confidence,
            self.scale_normalizer,
        ], dtype=np.float32)

    def to_dict(self) -> dict:
        return asdict(self)


def _euclidean_dist(p1: Tuple[float, float], p2: Tuple[float, float]) -> float:
    return math.hypot(p1[0] - p2[0], p1[1] - p2[1])


def _calculate_angle(a: Tuple[float, float], b: Tuple[float, float], c: Tuple[float, float]) -> float:
    """Calculate the angle ABC (at vertex B) in degrees [0, 180]."""
    v1 = (a[0] - b[0], a[1] - b[1])
    v2 = (c[0] - b[0], c[1] - b[1])
    mag1 = math.hypot(v1[0], v1[1])
    mag2 = math.hypot(v2[0], v2[1])
    if mag1 < 1e-5 or mag2 < 1e-5:
        return 180.0  # default straight arm if degenerate
    dot = v1[0] * v2[0] + v1[1] * v2[1]
    cosine = max(-1.0, min(1.0, dot / (mag1 * mag2)))
    return math.degrees(math.acos(cosine))


def extract_pose_features(
    keypoints: Union[Sequence[Tuple[float, float]], np.ndarray],
    confidences: Union[Sequence[float], np.ndarray],
    phone_box: Optional[Tuple[float, float, float, float]] = None,
    min_conf: float = 0.25,
    fallback_box: Optional[Tuple[float, float, float, float]] = None,
) -> PoseFeatureVector:
    """Extract normalized geometric pose features from raw 17 COCO keypoints.

    Args:
        keypoints: Array or sequence of (x, y) coordinates for 17 COCO keypoints.
        confidences: Array or sequence of confidence scores [0.0, 1.0].
        phone_box: Optional (x1, y1, x2, y2) phone bounding box.
        min_conf: Minimum confidence threshold to consider a keypoint visible.
        fallback_box: Optional bounding box for normalizer fallback.

    Returns:
        PoseFeatureVector with scale-normalized distances and angles.
    """
    kpts = list(keypoints)
    confs = [float(c) for c in confidences]

    def is_valid(idx: int) -> bool:
        return idx < len(kpts) and idx < len(confs) and confs[idx] >= min_conf

    def get_pt(idx: int) -> Optional[Tuple[float, float]]:
        return (float(kpts[idx][0]), float(kpts[idx][1])) if is_valid(idx) else None

    # Determine Scale Normalizer
    # Priority 1: Shoulder width
    l_sh = get_pt(LEFT_SHOULDER)
    r_sh = get_pt(RIGHT_SHOULDER)
    scale = None

    if l_sh and r_sh:
        sh_width = _euclidean_dist(l_sh, r_sh)
        if sh_width >= 10.0:
            scale = sh_width

    # Priority 2: Torso diagonal (shoulder midpoint to hip midpoint)
    if scale is None:
        l_hip = get_pt(LEFT_HIP)
        r_hip = get_pt(RIGHT_HIP)
        if l_sh and l_hip:
            scale = _euclidean_dist(l_sh, l_hip)
        elif r_sh and r_hip:
            scale = _euclidean_dist(r_sh, r_hip)

    # Priority 3: Fallback box diagonal or default
    if scale is None or scale < 10.0:
        if fallback_box is not None:
            w = fallback_box[2] - fallback_box[0]
            h = fallback_box[3] - fallback_box[1]
            scale = max(math.hypot(w, h) * 0.4, 10.0)
        else:
            scale = 100.0  # safe default pixel scale

    # Helper for normalized distance (returns 5.0 as large missing default)
    def norm_dist(idx1: int, idx2: int) -> float:
        p1, p2 = get_pt(idx1), get_pt(idx2)
        if p1 is None or p2 is None:
            return 5.0
        return min(5.0, _euclidean_dist(p1, p2) / scale)

    # Wrist to Ear / Nose / Shoulder distances
    lw_le = norm_dist(LEFT_WRIST, LEFT_EAR)
    lw_re = norm_dist(LEFT_WRIST, RIGHT_EAR)
    rw_re = norm_dist(RIGHT_WRIST, RIGHT_EAR)
    rw_le = norm_dist(RIGHT_WRIST, LEFT_EAR)

    lw_nose = norm_dist(LEFT_WRIST, NOSE)
    rw_nose = norm_dist(RIGHT_WRIST, NOSE)

    lw_ls = norm_dist(LEFT_WRIST, LEFT_SHOULDER)
    rw_rs = norm_dist(RIGHT_WRIST, RIGHT_SHOULDER)

    # Elbow Angles
    # Left arm: Shoulder(5) -> Elbow(7) -> Wrist(9)
    if is_valid(LEFT_SHOULDER) and is_valid(LEFT_ELBOW) and is_valid(LEFT_WRIST):
        l_angle = _calculate_angle(get_pt(LEFT_SHOULDER), get_pt(LEFT_ELBOW), get_pt(LEFT_WRIST))
    else:
        l_angle = 180.0

    # Right arm: Shoulder(6) -> Elbow(8) -> Wrist(10)
    if is_valid(RIGHT_SHOULDER) and is_valid(RIGHT_ELBOW) and is_valid(RIGHT_WRIST):
        r_angle = _calculate_angle(get_pt(RIGHT_SHOULDER), get_pt(RIGHT_ELBOW), get_pt(RIGHT_WRIST))
    else:
        r_angle = 180.0

    # Phone proximity features
    if phone_box is not None:
        ph_cx = (phone_box[0] + phone_box[2]) / 2.0
        ph_cy = (phone_box[1] + phone_box[3]) / 2.0
        ph_pt = (ph_cx, ph_cy)

        # Phone to wrists
        p_lw = get_pt(LEFT_WRIST)
        ph_lw = min(5.0, _euclidean_dist(ph_pt, p_lw) / scale) if p_lw else 5.0

        p_rw = get_pt(RIGHT_WRIST)
        ph_rw = min(5.0, _euclidean_dist(ph_pt, p_rw) / scale) if p_rw else 5.0

        # Phone to face (minimum distance to nose, left ear, right ear)
        face_pts = [get_pt(idx) for idx in (NOSE, LEFT_EAR, RIGHT_EAR) if is_valid(idx)]
        if face_pts:
            ph_face = min(5.0, min(_euclidean_dist(ph_pt, fp) for fp in face_pts) / scale)
        else:
            ph_face = 5.0
    else:
        ph_lw = 5.0
        ph_rw = 5.0
        ph_face = 5.0

    # Keypoint counts and mean confidence
    valid_confs = [c for c in confs if c >= min_conf]
    vis_count = float(len(valid_confs))
    mean_conf = float(sum(valid_confs) / len(valid_confs)) if valid_confs else 0.0

    return PoseFeatureVector(
        left_wrist_to_left_ear=float(lw_le),
        left_wrist_to_right_ear=float(lw_re),
        right_wrist_to_right_ear=float(rw_re),
        right_wrist_to_left_ear=float(rw_le),
        left_wrist_to_nose=float(lw_nose),
        right_wrist_to_nose=float(rw_nose),
        left_wrist_to_left_shoulder=float(lw_ls),
        right_wrist_to_right_shoulder=float(rw_rs),
        left_elbow_angle_deg=float(l_angle),
        right_elbow_angle_deg=float(r_angle),
        phone_to_left_wrist=float(ph_lw),
        phone_to_right_wrist=float(ph_rw),
        phone_to_face=float(ph_face),
        visible_keypoint_count=vis_count,
        pose_confidence=mean_conf,
        scale_normalizer=float(scale),
    )
