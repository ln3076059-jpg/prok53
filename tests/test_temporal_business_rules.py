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
