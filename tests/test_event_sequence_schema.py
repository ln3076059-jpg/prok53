import json
from pathlib import Path
import pytest
from jsonschema import validate, ValidationError

@pytest.fixture
def schema():
    schema_path = Path("datasets/schemas/v2_event_sequence_annotation.schema.json")
    with open(schema_path, "r", encoding="utf-8") as f:
        return json.load(f)

def create_base_record():
    return {
        "sequence_id": "seq1",
        "video_id": "vid1",
        "source_id": "src1",
        "fps": 30.0,
        "frame_count": 300,
        "start_time": "2026-09-04T12:00:00Z",
        "end_time": "2026-09-04T12:00:10Z",
        "occupants": [{"occupant_id": "occ1", "role": "driver"}],
        "events": [],
        "context": {},
        "review_provenance": {
            "reviewer_type": "AI",
            "status": "AI_REVIEWED_PROPOSAL",
            "annotation_version": "v1.0"
        }
    }

def test_ai_proposal_is_valid(schema):
    record = create_base_record()
    validate(instance=record, schema=schema)

def test_ai_reviewer_cannot_be_human_approved(schema):
    record = create_base_record()
    record["review_provenance"]["status"] = "HUMAN_APPROVED"
    with pytest.raises(ValidationError):
        validate(instance=record, schema=schema)

def test_human_approval_requires_provenance(schema):
    record = create_base_record()
    record["review_provenance"] = {
        "reviewer_type": "HUMAN",
        "status": "HUMAN_APPROVED",
        "reviewer_id": "rev1",
        "reviewed_at": "2026-09-04T12:05:00Z",
        "evidence_hash": "hash123",
        "annotation_version": "v1.0"
    }
    validate(instance=record, schema=schema)

def test_human_approval_missing_provenance_fails(schema):
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
        validate(instance=record, schema=schema)

@pytest.mark.parametrize("valid_role", [
    "driver", "front_passenger", "rear_left", "rear_center", "rear_right", "unknown"
])
def test_valid_occupant_roles(schema, valid_role):
    record = create_base_record()
    record["occupants"][0]["role"] = valid_role
    validate(instance=record, schema=schema)

@pytest.mark.parametrize("invalid_role", ["passenger", "outside_person"])
def test_invalid_occupant_roles(schema, invalid_role):
    record = create_base_record()
    record["occupants"][0]["role"] = invalid_role
    with pytest.raises(ValidationError):
        validate(instance=record, schema=schema)

def test_phone_event_with_valid_label(schema):
    record = create_base_record()
    record["events"].append({
        "event_type": "PHONE",
        "occupant_id": "occ1",
        "start_frame": 0,
        "end_frame": 10,
        "label": "PHONE_USE"
    })
    validate(instance=record, schema=schema)

def test_phone_event_with_seatbelt_label_fails(schema):
    record = create_base_record()
    record["events"].append({
        "event_type": "PHONE",
        "occupant_id": "occ1",
        "start_frame": 0,
        "end_frame": 10,
        "label": "UNFASTENED"
    })
    with pytest.raises(ValidationError):
        validate(instance=record, schema=schema)

def test_seatbelt_event_with_valid_label(schema):
    record = create_base_record()
    record["events"].append({
        "event_type": "SEATBELT",
        "occupant_id": "occ1",
        "start_frame": 0,
        "end_frame": 10,
        "label": "UNFASTENED"
    })
    validate(instance=record, schema=schema)

def test_seatbelt_event_with_phone_label_fails(schema):
    record = create_base_record()
    record["events"].append({
        "event_type": "SEATBELT",
        "occupant_id": "occ1",
        "start_frame": 0,
        "end_frame": 10,
        "label": "PHONE_USE"
    })
    with pytest.raises(ValidationError):
        validate(instance=record, schema=schema)
