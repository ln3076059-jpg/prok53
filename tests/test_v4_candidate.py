from pathlib import Path

import pytest

from backend.ai.auxiliary import PhoneContext, PoseEvidence, reason_phone_context
from backend.ai.detector import NormalizedDetection, SafetyDetector
from backend.ai.events import Observation, TemporalEventEngine, TemporalState


def _phone(confidence: float = 0.8) -> NormalizedDetection:
    return NormalizedDetection(0, "phone", confidence, (40.0, 40.0, 60.0, 60.0), 7)


def _pose(hand=(50.0, 50.0), face=(90.0, 90.0)) -> PoseEvidence:
    return PoseEvidence(0.9, (face,), (hand,), (0.0, 0.0, 100.0, 100.0))


def test_v4_phone_context_taxonomy_separates_visibility_from_use():
    assert (
        reason_phone_context(None, None, occupant_confidence=1.0).context == PhoneContext.NO_PHONE
    )
    assert (
        reason_phone_context(_phone(), None, occupant_confidence=1.0).context
        == PhoneContext.UNKNOWN_CONTEXT
    )
    assert (
        reason_phone_context(_phone(), _pose(), occupant_confidence=1.0).context
        == PhoneContext.HANDHELD_PHONE_USE
    )
    assert (
        reason_phone_context(_phone(), _pose(hand=(78.0, 78.0)), occupant_confidence=1.0).context
        == PhoneContext.POSSIBLE_PHONE_USE
    )
    assert (
        reason_phone_context(
            _phone(), _pose(hand=(100.0, 100.0), face=(100.0, 0.0)), occupant_confidence=1.0
        ).context
        == PhoneContext.PHONE_VISIBLE_NOT_USED
    )
    assert (
        reason_phone_context(
            _phone(), _pose(), occupant_confidence=1.0, mounted_evidence=True
        ).context
        == PhoneContext.MOUNTED_PHONE
    )


def test_v4_strict_context_abstains_when_pose_or_assignment_is_unknown():
    engine = TemporalEventEngine(
        strict_phone_context=True,
        min_observations=1,
        min_positive_seconds=0,
        cooldown_seconds=0,
    )
    observation = Observation(
        0.0,
        "phone",
        0.9,
        1,
        "driver",
        "vehicle",
        phone_context=PhoneContext.UNKNOWN_CONTEXT.value,
    )
    assert engine.add(observation) == []
    assert engine.states[("vehicle", 1, "driver")] == TemporalState.NEEDS_REVIEW


def test_v4_stable_occupant_identity_survives_track_id_change_and_deduplicates():
    engine = TemporalEventEngine(
        strict_phone_context=True,
        window_seconds=2.0,
        min_positive_seconds=0.5,
        min_observations=2,
        cooldown_seconds=0,
        feature_positive_score=0.5,
        positive_ratio=0.5,
        candidate_threshold=0.5,
        activation_threshold=0.5,
        release_threshold=0.2,
    )

    def observation(timestamp: float, track_id: int, context: PhoneContext) -> Observation:
        return Observation(
            timestamp,
            "phone",
            0.9,
            track_id,
            "driver",
            "vehicle",
            phone_context=context.value,
            occupant_id="occupant-1",
        )

    assert engine.add(observation(0.0, 10, PhoneContext.HANDHELD_PHONE_USE)) == []
    events = engine.add(observation(0.6, 11, PhoneContext.HANDHELD_PHONE_USE))
    assert len(events) == 1
    assert events[0].review_status == "PENDING"
    assert engine.add(observation(1.0, 12, PhoneContext.HANDHELD_PHONE_USE)) == []
    key = ("vehicle", None, "occupant-1")
    assert engine.states[key] == TemporalState.ACTIVE

    assert engine.add(observation(1.2, 12, PhoneContext.PHONE_VISIBLE_NOT_USED)) == []
    assert engine.states[key] == TemporalState.CLOSED


def test_v4_possible_phone_use_is_review_only_not_auto_pending():
    engine = TemporalEventEngine(
        strict_phone_context=True,
        min_observations=1,
        min_positive_seconds=0,
        cooldown_seconds=0,
        feature_positive_score=0.5,
        positive_ratio=0.5,
        candidate_threshold=0.5,
        activation_threshold=0.5,
    )
    events = engine.add(
        Observation(
            0.0,
            "phone",
            0.9,
            1,
            "driver",
            "vehicle",
            phone_context=PhoneContext.POSSIBLE_PHONE_USE.value,
        )
    )
    assert len(events) == 1
    assert events[0].review_status == "NEEDS_REVIEW"


def test_v4_config_is_structurally_valid_and_production_blocked():
    detector = SafetyDetector(
        Path("models/active/best.pt"), Path("models/model_config_v4_candidate.yaml")
    )
    assert detector.validate_runtime("testing")["status"] == "STRUCTURALLY_VALID"
    assert detector.model_version == "ROADWATCH_V4_CANDIDATE_001"
    with pytest.raises(RuntimeError, match="production runtime validation failed"):
        detector.validate_runtime("production")
