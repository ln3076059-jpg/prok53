import pytest
from backend.ai.events import TemporalEventEngine, Observation

def make_engine():
    return TemporalEventEngine(
        window_seconds=2.0,
        min_positive_seconds=0.0,  # make tests easier
        min_observations=1,
        cooldown_seconds=0.0
    )

def test_passenger_phone_cannot_trigger_violation():
    engine = make_engine()
    obs = Observation(
        timestamp=1.0,
        class_name="phone",
        confidence=0.9,
        track_id=1,
        occupant_role="passenger",
        vehicle_context_id="veh1",
        phone_context="HANDHELD_USE"
    )
    events = engine.add(obs)
    assert not events, "Passenger phone MUST NOT trigger an event"

def test_unknown_role_phone_cannot_emit_event():
    engine = make_engine()
    obs = Observation(
        timestamp=1.0,
        class_name="phone",
        confidence=0.9,
        track_id=1,
        occupant_role="unknown",
        vehicle_context_id="veh1",
        phone_context="HANDHELD_USE"
    )
    events = engine.add(obs)
    assert not events, "Unknown role MUST NOT emit a PHONE event"

def test_outside_person_cannot_trigger_violation():
    engine = make_engine()
    obs = Observation(
        timestamp=1.0,
        class_name="phone",
        confidence=0.9,
        track_id=1,
        occupant_role="driver",
        vehicle_context_id=None,  # No vehicle context -> outside
        phone_context="HANDHELD_USE"
    )
    events = engine.add(obs)
    assert not events, "Outside person MUST NOT trigger PHONE_USE"

def test_motorcycle_cannot_trigger_seatbelt_violation():
    engine = make_engine()
    obs = Observation(
        timestamp=1.0,
        class_name="seatbelt_unfastened",
        confidence=0.9,
        track_id=1,
        occupant_role="driver",
        vehicle_context_id="veh1",
        vehicle_type="motorcycle"
    )
    events = engine.add(obs)
    assert not events, "Motorcycles MUST NOT trigger NO_SEATBELT"

def test_unknown_belt_cannot_trigger_violation():
    engine = make_engine()
    obs = Observation(
        timestamp=1.0,
        class_name="uncertain_or_occluded",
        confidence=0.9,
        track_id=1,
        occupant_role="driver",
        vehicle_context_id="veh1"
    )
    events = engine.add(obs)
    assert not events, "UNKNOWN / UNCERTAIN belt MUST fail-closed"

def test_mounted_phone_cannot_trigger_violation():
    engine = make_engine()
    obs = Observation(
        timestamp=1.0,
        class_name="phone",
        confidence=0.9,
        track_id=1,
        occupant_role="driver",
        vehicle_context_id="veh1",
        phone_context="MOUNTED_OR_STATIC"
    )
    events = engine.add(obs)
    assert not events, "Mounted/static phone MUST NOT trigger PHONE_USE"

def test_single_frame_evidence_cannot_trigger_violation():
    engine = TemporalEventEngine(
        window_seconds=2.0,
        min_positive_seconds=0.0,
        min_observations=5,  # Requires 5 frames
        cooldown_seconds=0.0
    )
    obs = Observation(
        timestamp=1.0,
        class_name="phone",
        confidence=0.9,
        track_id=1,
        occupant_role="driver",
        vehicle_context_id="veh1",
        phone_context="HANDHELD_USE"
    )
    events = engine.add(obs)
    assert not events, "Single-frame spike MUST NOT trigger an event violation"


def test_rejected_driver_observation_releases_active_phone_event():
    engine = TemporalEventEngine(
        window_seconds=2.0,
        min_positive_seconds=0.0,
        min_observations=2,
        cooldown_seconds=0.0,
        feature_positive_score=0.2,
        candidate_threshold=0.2,
        activation_threshold=0.2,
        release_threshold=0.15,
    )
    for timestamp in (1.0, 1.5):
        engine.add(Observation(
            timestamp=timestamp,
            class_name="phone",
            confidence=0.9,
            fusion_score=0.9,
            track_id=1,
            occupant_role="driver",
            vehicle_context_id="veh1",
            phone_context="HANDHELD_USE",
        ))
    assert engine.active_events == {("PHONE", "veh1", 1, "driver")}

    engine.add(Observation(
        timestamp=2.0,
        class_name="phone",
        confidence=0.0,
        fusion_score=0.0,
        track_id=1,
        occupant_role="driver",
        vehicle_context_id="veh1",
        phone_context="UNKNOWN",
        evidence_source="AUTONOMOUS_V21_REJECTED",
    ))
    assert not engine.active_events
