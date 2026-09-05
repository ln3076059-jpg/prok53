import pytest
from training.validate_event_sequence_annotations import validate_annotation

def create_valid_annotation():
    return {
        "sequence_id": "seq1",
        "video_id": "vid1",
        "source_id": "src1",
        "fps": 30.0,
        "frame_count": 300,
        "start_time": "2026-09-04T12:00:00Z",
        "end_time": "2026-09-04T12:00:10Z",
        "occupants": [
            {"occupant_id": "occ1", "role": "driver", "role_confidence": 0.9}
        ],
        "events": [
            {
                "event_type": "PHONE",
                "occupant_id": "occ1",
                "start_frame": 30,
                "end_frame": 60,
                "start_time_sec": 1.0,
                "end_time_sec": 2.0,
                "label": "PHONE_USE"
            }
        ],
        "context": {},
        "review_provenance": {
            "reviewer_type": "HUMAN",
            "status": "HUMAN_APPROVED",
            "reviewer_id": "rev1",
            "reviewed_at": "2026-09-04T12:05:00Z",
            "evidence_hash": "hash123",
            "annotation_version": "v1.0"
        }
    }

def test_valid_annotation():
    ann = create_valid_annotation()
    errors = validate_annotation(ann, {})
    # Ignoring schema validation errors by passing {} schema since the validator checks schema errors first
    # Wait, the Draft7Validator with {} schema will always pass.
    # We only care about the semantic errors (those prefixed with 'Semantic error:')
    semantic_errors = [e for e in errors if e.startswith('Semantic error:')]
    assert not semantic_errors, f"Expected no semantic errors, got {semantic_errors}"

def test_duplicate_occupant_id():
    ann = create_valid_annotation()
    ann["occupants"].append({"occupant_id": "occ1", "role": "front_passenger"})
    errors = validate_annotation(ann, {})
    semantic_errors = [e for e in errors if e.startswith('Semantic error:')]
    assert any("duplicate occupant_id" in e for e in semantic_errors)

def test_unknown_occupant_id_in_event():
    ann = create_valid_annotation()
    ann["events"][0]["occupant_id"] = "occ_unknown"
    errors = validate_annotation(ann, {})
    semantic_errors = [e for e in errors if e.startswith('Semantic error:')]
    assert any("not found in occupants" in e for e in semantic_errors)

def test_start_frame_greater_than_end_frame():
    ann = create_valid_annotation()
    ann["events"][0]["start_frame"] = 90
    ann["events"][0]["end_frame"] = 60
    errors = validate_annotation(ann, {})
    semantic_errors = [e for e in errors if e.startswith('Semantic error:')]
    assert any("start_frame 90 > end_frame 60" in e for e in semantic_errors)

def test_end_frame_greater_equal_frame_count():
    ann = create_valid_annotation()
    ann["events"][0]["end_frame"] = 300  # frame_count is 300
    errors = validate_annotation(ann, {})
    semantic_errors = [e for e in errors if e.startswith('Semantic error:')]
    assert any("end_frame 300 >= frame_count 300" in e for e in semantic_errors)

def test_start_time_greater_than_end_time():
    ann = create_valid_annotation()
    ann["events"][0]["start_time_sec"] = 3.0
    ann["events"][0]["end_time_sec"] = 2.0
    errors = validate_annotation(ann, {})
    semantic_errors = [e for e in errors if e.startswith('Semantic error:')]
    assert any("start_time 3.0 > end_time 2.0" in e for e in semantic_errors)

def test_end_time_greater_than_duration():
    ann = create_valid_annotation()
    # duration is frame_count / fps = 300/30 = 10s
    ann["events"][0]["end_time_sec"] = 11.0
    ann["events"][0]["end_frame"] = 60 # avoid frame bounds vs time bounds error for now, or just let it fail multiple times
    errors = validate_annotation(ann, {})
    semantic_errors = [e for e in errors if e.startswith('Semantic error:')]
    assert any("end_time_sec 11.0 > duration 10.0" in e for e in semantic_errors)

def test_sequence_start_time_greater_than_end_time():
    ann = create_valid_annotation()
    ann["start_time"] = "2026-09-04T12:00:20Z"
    ann["end_time"] = "2026-09-04T12:00:10Z"
    errors = validate_annotation(ann, {})
    semantic_errors = [e for e in errors if e.startswith('Semantic error:')]
    assert any("sequence start_time" in e and "must be < end_time" in e for e in semantic_errors)

def test_frame_time_boundaries_mismatch():
    ann = create_valid_annotation()
    ann["events"][0]["start_time_sec"] = 2.0 # 30 frames @ 30fps is 1.0s
    errors = validate_annotation(ann, {})
    semantic_errors = [e for e in errors if e.startswith('Semantic error:')]
    assert any("does not match start_frame" in e for e in semantic_errors)

def test_ai_proposal_not_valid():
    ann = create_valid_annotation()
    ann["review_provenance"]["status"] = "AI_REVIEWED_PROPOSAL"
    errors = validate_annotation(ann, {})
    semantic_errors = [e for e in errors if e.startswith('Semantic error:')]
    assert any("AI proposals are not valid for governed calibration" in e for e in semantic_errors)

def test_contradictory_context():
    ann = create_valid_annotation()
    ann["context"]["inside_vehicle"] = True
    ann["context"]["outside_vehicle_person"] = True
    errors = validate_annotation(ann, {})
    semantic_errors = [e for e in errors if e.startswith('Semantic error:')]
    assert any("contradictory context - inside_vehicle and outside_vehicle_person cannot both be True" in e for e in semantic_errors)
