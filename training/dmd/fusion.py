"""Multi-View Feature Fusion Engine for Roadwatch V3.

Fuses evidence across BODY, FACE, and HANDS streams with Pose Features and Temporal Context:
- BODY: Primary cabin context, driver assignment, torso/arm geometry, center camera phone detections.
- FACE: Resolves left-ear head/shoulder occlusions via close-up face/ear stream.
- HANDS: Resolves lap/steering wheel occlusions via close-up steering/lap stream.
- POSE: Normalized geometric features (wrist-to-ear, elbow angles, wrist-to-phone).
- TEMPORAL: Rolling window persistence (15, 20, 30 frames) suppressing transient false alarms.

Adheres strictly to Fail-Closed Governance:
- Exposes explicit view availability flags.
- If BODY view is absent, returns NEEDS_REVIEW (cannot verify driver vs passenger).
- Never infers physical phone presence without physical visual detection evidence in at least one view.
- Suppresses scratching/hair grooming via explicit non-distraction gesture scoring.
"""
from __future__ import annotations

import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.dmd.action_model import ActionProbabilities, RuleGuidedPoseClassifier, TemporalWindowAggregator
from training.dmd.pose_features import PoseFeatureVector, extract_pose_features


@dataclass(frozen=True)
class FusedPhoneObservation:
    timestamp_seconds: float
    frame_index: int
    fused_score: float
    action_type: str  # "NORMAL", "PHONECALL_RIGHT", "PHONECALL_LEFT", "TEXTING", "UNKNOWN"
    phone_detected: bool
    governance_status: str  # "CONFIRMED", "FAIL_CLOSED_NO_PHONE", "NEEDS_REVIEW"

    # Per-component scores
    body_phone_conf: float
    face_phone_conf: float
    hands_phone_conf: float
    pose_call_score: float
    pose_text_score: float
    temporal_persistence: float

    # View availability
    body_available: bool
    face_available: bool
    hands_available: bool

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class MultiViewFusionEngine:
    """Combines multi-view detections, pose features, and temporal context into a unified evidence score."""

    def __init__(
        self,
        phone_conf_threshold: float = 0.25,
        temporal_window: int = 30,
        enable_face_fusion: bool = True,
        enable_hands_fusion: bool = True,
        enable_pose_fusion: bool = True,
    ):
        self.phone_conf_threshold = phone_conf_threshold
        self.enable_face_fusion = enable_face_fusion
        self.enable_hands_fusion = enable_hands_fusion
        self.enable_pose_fusion = enable_pose_fusion

        self.pose_classifier = RuleGuidedPoseClassifier()
        self.temporal_aggregator = TemporalWindowAggregator(window_size=temporal_window)

    def reset(self) -> None:
        self.temporal_aggregator.reset()

    def fuse_frame(
        self,
        timestamp_seconds: float,
        frame_index: int,
        body_phone_conf: float = 0.0,
        face_phone_conf: float = 0.0,
        hands_phone_conf: float = 0.0,
        body_pose_features: Optional[PoseFeatureVector] = None,
        body_available: bool = True,
        face_available: bool = False,
        hands_available: bool = False,
    ) -> FusedPhoneObservation:
        # 1. Fail-closed check: BODY is mandatory for primary cabin association
        if not body_available:
            return FusedPhoneObservation(
                timestamp_seconds=timestamp_seconds,
                frame_index=frame_index,
                fused_score=0.0,
                action_type="UNKNOWN",
                phone_detected=False,
                governance_status="NEEDS_REVIEW",
                body_phone_conf=0.0,
                face_phone_conf=0.0,
                hands_phone_conf=0.0,
                pose_call_score=0.0,
                pose_text_score=0.0,
                temporal_persistence=0.0,
                body_available=False,
                face_available=face_available,
                hands_available=hands_available,
            )

        # 2. Multi-view phone detection evidence fusion
        # Effective view confidences gated by enabled flags and availability
        eff_face = face_phone_conf if (self.enable_face_fusion and face_available) else 0.0
        eff_hands = hands_phone_conf if (self.enable_hands_fusion and hands_available) else 0.0
        eff_body = body_phone_conf

        # Highest detection confidence across active views
        # Multi-view insight: phone can be occluded in BODY but visible in FACE (e.g. left ear) or HANDS (lap)
        max_view_phone_conf = max(eff_body, eff_face * 0.90, eff_hands * 0.90)

        # 3. Pose features and Action classification
        pose_arr = body_pose_features.to_array() if (self.enable_pose_fusion and body_pose_features) else None
        probs = self.pose_classifier.predict_probabilities(pose_arr, phone_detector_conf=max_view_phone_conf)

        # 4. Temporal window update
        self.temporal_aggregator.push(probs, phone_conf=max_view_phone_conf)
        temporal_stats = self.temporal_aggregator.aggregate()

        # 5. Determine specific action type
        action_type = "NORMAL"
        temporal_persist = 0.0

        if probs.phonecall > 0.30 or eff_face > 0.35:
            # Check left vs right from pose features if available
            if body_pose_features and (body_pose_features.left_wrist_to_left_ear < body_pose_features.right_wrist_to_right_ear):
                action_type = "PHONECALL_LEFT"
            else:
                action_type = "PHONECALL_RIGHT"
            temporal_persist = temporal_stats["call_persistence"]

        elif probs.texting > 0.30 or eff_hands > 0.35:
            action_type = "TEXTING"
            temporal_persist = temporal_stats["text_persistence"]

        # 6. Fused evidence score calculation
        # Base: visual phone detection evidence
        # Boost: consistent pose + temporal persistence
        # Penalty: lack of physical phone (cannot invent phone without physical visual confirmation)
        if max_view_phone_conf < 0.10:
            # Section 25: Never infer a physically invisible phone solely because system expects action
            fused_score = max_view_phone_conf * 0.5
            governance_status = "FAIL_CLOSED_NO_PHONE"
            phone_detected = False
            action_type = "NORMAL"
        else:
            # Phone is visible in at least one view:
            # Combine multi-view detector score with pose and temporal support
            action_support = max(probs.phonecall, probs.texting)
            fused_score = min(
                1.0,
                0.55 * max_view_phone_conf
                + 0.25 * action_support
                + 0.20 * temporal_persist
            )
            phone_detected = fused_score >= self.phone_conf_threshold
            governance_status = "CONFIRMED" if phone_detected else "FAIL_CLOSED_LOW_CONF"

        return FusedPhoneObservation(
            timestamp_seconds=round(timestamp_seconds, 3),
            frame_index=frame_index,
            fused_score=round(float(fused_score), 4),
            action_type=action_type,
            phone_detected=phone_detected,
            governance_status=governance_status,
            body_phone_conf=round(float(eff_body), 4),
            face_phone_conf=round(float(eff_face), 4),
            hands_phone_conf=round(float(eff_hands), 4),
            pose_call_score=round(float(probs.phonecall), 4),
            pose_text_score=round(float(probs.texting), 4),
            temporal_persistence=round(float(temporal_persist), 4),
            body_available=body_available,
            face_available=face_available,
            hands_available=hands_available,
        )
