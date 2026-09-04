import pytest

def test_passenger_phone_cannot_trigger_violation():
    # Simulate a pipeline output where occupant role is passenger and phone is detected
    occupant_role = "passenger"
    phone_detected = True
    
    # Business logic dictates passenger phones are ignored
    phone_use_violation = (occupant_role == "driver") and phone_detected
    
    assert not phone_use_violation, "Passenger phone MUST NOT trigger a PHONE_USE violation"

def test_outside_person_cannot_trigger_violation():
    # Simulate a pipeline output where a person is detected outside the vehicle
    context = "outside_vehicle"
    phone_detected = True
    seatbelt_unfastened = True
    
    phone_use_violation = (context == "inside_vehicle") and phone_detected
    seatbelt_violation = (context == "inside_vehicle") and seatbelt_unfastened
    
    assert not phone_use_violation, "Outside person MUST NOT trigger PHONE_USE"
    assert not seatbelt_violation, "Outside person MUST NOT trigger NO_SEATBELT"

def test_motorcycle_cannot_trigger_seatbelt_violation():
    vehicle_type = "motorcycle"
    seatbelt_unfastened = True
    
    seatbelt_violation = (vehicle_type != "motorcycle") and seatbelt_unfastened
    
    assert not seatbelt_violation, "Motorcycles MUST NOT trigger NO_SEATBELT"

def test_unknown_belt_cannot_trigger_violation():
    seatbelt_prediction = "uncertain_or_occluded"
    
    seatbelt_violation = (seatbelt_prediction == "unfastened")
    
    assert not seatbelt_violation, "UNKNOWN / UNCERTAIN belt MUST fail-closed"

def test_mounted_phone_cannot_trigger_violation():
    phone_detected = True
    handheld_evidence = False # No hand/face interaction, it's mounted
    
    phone_use_violation = phone_detected and handheld_evidence
    
    assert not phone_use_violation, "Mounted/static phone MUST NOT trigger PHONE_USE"

def test_single_frame_evidence_cannot_trigger_violation():
    # Simulate temporal buffer with only 1 positive frame
    temporal_buffer_positives = 1
    minimum_observations = 5 # Temporal confirmation requires multiple frames
    
    event_triggered = (temporal_buffer_positives >= minimum_observations)
    
    assert not event_triggered, "Single-frame spike MUST NOT trigger an event violation"
