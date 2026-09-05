import json
from pathlib import Path
import pytest
from jsonschema import Draft7Validator, FormatChecker, ValidationError

@pytest.fixture
def schema():
    schema_path = Path("datasets/schemas/v2_event_sequence_annotation.schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

@pytest.fixture
def validator(schema):
    return Draft7Validator(schema, format_checker=FormatChecker())

def create_base_record():
    return {
        "sequence_id": "seq1",
        "video_id": "vid1",
        "vehicle_id": "video:vid1:vehicle-track:1",
        "cabin_id": "video:vid1:vehicle-track:1:cabin:1",
        "source_id": "src1",
        "fps": 30.0,
        "frame_count": 300,
        "start_time": "2026-09-04T12:00:00Z",
        "end_time": "2026-09-04T12:00:10Z",
        "occupants": [{"occupant_id": "video:vid1:vehicle-track:1:cabin:1:occupant-track:1", "role": "driver"}],
        "events": [],
        "context_intervals": [
            {
                "context_id": "ctx1",
                "occupant_id": "video:vid1:vehicle-track:1:cabin:1:occupant-track:1",
                "start_frame": 0,
                "end_frame": 300,
                "inside_vehicle": True,
                "outside_vehicle_person": False,
                "motorcycle_flag": False,
                "phone_state": "NO_PHONE",
                "seatbelt_state": "FASTENED",
                "visibility": "clear",
                "conditions": "daylight",
            }
        ],
        "context": {},
        "review_provenance": {
            "reviewer_type": "AI",
            "status": "AI_REVIEWED_PROPOSAL",
            "annotation_version": "v1.0"
        }
    }

def test_ai_proposal_is_valid(validator):
    record = create_base_record()
    validator.validate(record)

def test_ai_reviewer_cannot_be_human_approved(validator):
    record = create_base_record()
    record["review_provenance"]["status"] = "HUMAN_APPROVED"
    with pytest.raises(ValidationError):
        validator.validate(record)

def test_human_approval_requires_provenance(validator):
    record = create_base_record()
    record["review_provenance"] = {
        "reviewer_type": "HUMAN",
        "status": "HUMAN_APPROVED",
        "reviewer_id": "rev1",
        "reviewed_at": "2026-09-04T12:05:00Z",
        "evidence_hash": "hash123",
        "annotation_version": "v1.0"
    }
    validator.validate(record)

def test_human_approval_missing_provenance_fails(validator):
    record = create_base_record()
    # Missing reviewer_id
    record["review_provenance"] = {
        "reviewer_type": "HUMAN",
        "status": "HUMAN_APPROVED",
        "reviewed_at": "2026-09-04T12:05:00Z",
        "evidence_hash": "hash123",
        "annotation_version": "v1.0"
    }
    with pytest.raises(ValidationError):
        validator.validate(record)

def test_invalid_date_time_format_fails(validator):
    record = create_base_record()
    record["start_time"] = "yesterday"
    with pytest.raises(ValidationError):
        validator.validate(record)
        
def test_invalid_reviewed_at_date_time_format_fails(validator):
    record = create_base_record()
    record["review_provenance"] = {
        "reviewer_type": "HUMAN",
        "status": "HUMAN_APPROVED",
        "reviewer_id": "rev1",
        "reviewed_at": "abc",
        "evidence_hash": "hash123",
        "annotation_version": "v1.0"
    }
    with pytest.raises(ValidationError):
        validator.validate(record)

@pytest.mark.parametrize("valid_role", [
    "driver", "front_passenger", "rear_left", "rear_center", "rear_right", "unknown"
])
def test_valid_occupant_roles(validator, valid_role):
    record = create_base_record()
    record["occupants"][0]["role"] = valid_role
    validator.validate(record)

@pytest.mark.parametrize("invalid_role", ["passenger", "outside_person"])
def test_invalid_occupant_roles(validator, invalid_role):
    record = create_base_record()
    record["occupants"][0]["role"] = invalid_role
    with pytest.raises(ValidationError):
        validator.validate(record)

def test_phone_event_with_valid_label(validator):
    record = create_base_record()
    record["events"].append({
        "event_type": "PHONE",
        "occupant_id": "video:vid1:vehicle-track:1:cabin:1:occupant-track:1",
        "start_frame": 0,
        "end_frame": 10,
        "label": "PHONE_USE"
    })
    validator.validate(record)

def test_phone_event_with_seatbelt_label_fails(validator):
    record = create_base_record()
    record["events"].append({
        "event_type": "PHONE",
        "occupant_id": "video:vid1:vehicle-track:1:cabin:1:occupant-track:1",
        "start_frame": 0,
        "end_frame": 10,
        "label": "UNFASTENED"
    })
    with pytest.raises(ValidationError):
        validator.validate(record)

def test_seatbelt_event_with_valid_label(validator):
    record = create_base_record()
    record["events"].append({
        "event_type": "SEATBELT",
        "occupant_id": "video:vid1:vehicle-track:1:cabin:1:occupant-track:1",
        "start_frame": 0,
        "end_frame": 10,
        "label": "UNFASTENED"
    })
    validator.validate(record)

def test_seatbelt_event_with_phone_label_fails(validator):
    record = create_base_record()
    record["events"].append({
        "event_type": "SEATBELT",
        "occupant_id": "video:vid1:vehicle-track:1:cabin:1:occupant-track:1",
        "start_frame": 0,
        "end_frame": 10,
        "label": "PHONE_USE"
    })
    with pytest.raises(ValidationError):
        validator.validate(record)
