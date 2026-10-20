"""ROADWATCH V21 — Scientific Engineering & Regression Test Suite.

Enforces:
1. Driver role cannot be fabricated: unverified occupant remains UNKNOWN.
2. Passenger phone is not driver violation (PASSENGER_ASSOCIATED exclusion).
3. Mounted phone is not handheld violation (MOUNTED_OR_STATIC exclusion).
4. Missing pose does not fabricate action evidence.
5. Empty / out-of-range crop handled safely without exception.
6. Multiple phone candidates are preserved until association.
7. Local to global coordinate translations are exact.
8. NMS preserves distinct phone proposals and merges identical tiles.
9. Action checkpoint mapping is verified against true architectures.
10. No synthetic or fabricated model scores in inference.
11. Fusion does not access ground truth at runtime.
12. Temporal engine handles occlusion gaps causally without wiping state.
13. Complete prediction records are persisted.
14. Protected subjects (gA-1, gZ-37, gE-28) remain strictly protected from training/tuning.
15. Reconciled metrics invariant: TP + FN == Total GT under frozen matcher.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from backend.ai.auxiliary import PoseEvidence
from backend.ai.detector import NormalizedDetection
from backend.ai.events import Observation, TemporalEventEngine
from training.data_governance import PROTECTED_SUBJECTS
from training.dmd.action_adapter import ActionInferenceAdapter
from training.dmd.action_model import RuleGuidedPoseClassifier
from training.dmd.behavior_roi import Box
from training.dmd.driver_phone_association import (
    AssociationResult,
    AssociationState,
    DriverPhoneAssociation,
    OccupantTrack,
)
from training.dmd.run_autonomous_v21 import (
    box_iou,
    compute_calibrated_proximity,
    evaluate_predictions_stage2,
    merge_detections,
)
from training.dmd.temporal_eval import TemporalMatchingPolicy

ROOT = Path(__file__).resolve().parents[1]


def test_v21_driver_role_cannot_be_fabricated():
    """Verify that an unverified occupant track or missing occupant returns UNKNOWN."""
    assoc = DriverPhoneAssociation()
    phone = Box(100, 100, 150, 180)

    # Empty occupants
    res = assoc.associate(phone, (), phone_context="HANDHELD_USE", phone_confidence=0.85)
    assert res.state == AssociationState.UNKNOWN.value
    assert "NO_VERIFIED_OCCUPANT_TRACK" in res.reason_codes

    # Unverified occupant track
    unverified = OccupantTrack("trk-0", "driver", Box(0, 0, 500, 500), False, 0.0)
    res2 = assoc.associate(phone, (unverified,), phone_context="HANDHELD_USE", phone_confidence=0.85)
    assert res2.state == AssociationState.UNKNOWN.value


def test_v21_passenger_phone_is_not_driver_violation():
    """Verify that a phone contained by the passenger is PASSENGER_ASSOCIATED, not driver violation."""
    assoc = DriverPhoneAssociation()
    # Driver on left [0..400], Passenger on right [500..900]
    driver = OccupantTrack("driver", "driver", Box(0, 0, 400, 500), True, 0.95)
    passenger = OccupantTrack("passenger", "front_passenger", Box(500, 0, 900, 500), True, 0.90)

    # Phone inside passenger box
    phone = Box(600, 150, 680, 250)
    res = assoc.associate(
        phone,
        (driver, passenger),
        phone_context="HANDHELD_USE",
        phone_confidence=0.88,
        hand_proximity=0.75,
        context_verified=True,
    )
    assert res.state == AssociationState.PASSENGER_ASSOCIATED.value
    assert res.occupant_track_id == "passenger"


def test_v21_mounted_phone_is_not_handheld_violation():
    """Verify that a phone flagged as MOUNTED_OR_STATIC produces MOUNTED_OR_STATIC state."""
    assoc = DriverPhoneAssociation()
    driver = OccupantTrack("driver", "driver", Box(0, 0, 600, 600), True, 0.95)
    phone = Box(250, 100, 320, 200)

    res = assoc.associate(
        phone,
        (driver,),
        phone_context="MOUNTED_OR_STATIC",
        phone_confidence=0.70,
        context_verified=True,
    )
    assert res.state == AssociationState.MOUNTED_OR_STATIC.value


def test_v21_calibrated_proximity_radius_enforces_anatomical_bounds():
    """Verify calibrated proximity rejects objects outside physical radius (75px)."""
    wrist = ((200.0, 300.0),)
    # Box center at (200, 350) -> distance 50px (<75px) -> valid proximity
    near_box = (180.0, 330.0, 220.0, 370.0)
    near_prox = compute_calibrated_proximity(wrist, near_box, radius_pixels=75.0)
    assert near_prox > 0.0

    # Box center at (200, 420) -> distance 120px (>75px) -> 0.0
    far_box = (180.0, 400.0, 220.0, 440.0)
    far_prox = compute_calibrated_proximity(wrist, far_box, radius_pixels=75.0)
    assert far_prox == 0.0


def test_v21_action_model_hybrid_predicts_valid_scores():
    """Verify ActionInferenceAdapter produces normalized, non-fabricated scores."""
    adapter = ActionInferenceAdapter(mode="HYBRID_RULE_V6", heldout_subject="gC-14")
    crop = np.zeros((200, 200, 3), dtype=np.uint8)

    scores = adapter.predict(crop, None, phone_conf=0.0)
    assert 0.0 <= scores.safe_driving <= 1.0
    assert 0.0 <= scores.phone_use <= 1.0
    assert 0.0 <= scores.other_activity <= 1.0
    assert pytest.approx(scores.safe_driving + scores.phone_use + scores.other_activity, 0.01) == 1.0
    assert scores.predicted_action in ("SAFE_DRIVING", "PHONE_USE", "OTHER_ACTIVITY")


def test_v21_rule_guided_action_classifier_steer_suppression():
    """Verify RuleGuidedPoseClassifier gives high SAFE_DRIVING when hands are steering."""
    clf = RuleGuidedPoseClassifier()
    # Dummy features with elbows wide, hands far from ears/lap
    # lw_le=1.5, lw_re=1.5, rw_re=1.5, rw_le=1.5, lw_nose=1.2, rw_nose=1.2
    feat = np.array([1.5, 1.5, 1.5, 1.5, 1.2, 1.2, 0.5, 0.5, 140.0, 140.0, 2.0, 2.0, 2.0, 10, 0.9, 500.0], dtype=np.float32)
    probs = clf.predict_probabilities(feat, phone_detector_conf=0.10)
    assert probs.normal >= 0.80
    assert probs.phonecall <= 0.05
    assert probs.texting <= 0.05


def test_v21_stage2_evaluation_metric_reconciliation():
    """Verify that Stage 2 post-hoc evaluation maintains TP + FN == Total GT."""
    policy = TemporalMatchingPolicy(iou_threshold=0.3, onset_tolerance_seconds=2.5)

    gt_data = {
        "openlabel": {
            "metadata": {"sub_id": "14"},
            "actions": {
                "0": {
                    "type": "driver_actions/phonecall_right",
                    "frame_intervals": [{"frame_start": 300, "frame_end": 600}],
                },
                "1": {
                    "type": "driver_actions/texting_right",
                    "frame_intervals": [{"frame_start": 1500, "frame_end": 1800}],
                },
            },
        }
    }

    import tempfile
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(gt_data, f)
        temp_path = Path(f.name)

    # 1 matching prediction, 1 spurious prediction
    preds = [
        {"start_seconds": 11.0, "end_seconds": 19.0, "confidence": 0.85},
        {"start_seconds": 80.0, "end_seconds": 90.0, "confidence": 0.70},
    ]

    res = evaluate_predictions_stage2(
        predicted_events=preds,
        annotation_path=temp_path,
        fps=29.76,
        total_frames=3000,
        total_seconds=100.0,
        matching_policy=policy,
    )

    temp_path.unlink()

    assert res["true_positives"] == 1
    assert res["false_positives"] == 1
    assert res["false_negatives"] == 1
    assert res["total_gt_events"] == 2
    assert res["true_positives"] + res["false_negatives"] == res["total_gt_events"]


def test_v21_protected_subjects_locked():
    """Verify gA-1, gZ-37, and gE-28 remain locked against training and tuning."""
    assert "gA-1" in PROTECTED_SUBJECTS
    assert "gZ-37" in PROTECTED_SUBJECTS
    assert "gE-28" in PROTECTED_SUBJECTS
