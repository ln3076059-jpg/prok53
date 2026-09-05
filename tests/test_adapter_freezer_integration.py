import json
import csv
from pathlib import Path
import tempfile
import pytest

from training.validate_event_sequence_annotations import load_schema
from training.build_event_truth_from_sequences import process_file
from training.freeze_event_ground_truth import freeze_event_ground_truth

def create_valid_external_lock(path: Path, video_id: str):
    lock_data = {
        "status": "FROZEN_EXTERNAL_TEST",
        "human_review_status": "ALL_APPROVED",
        "video_ids": [video_id]
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lock_data, f)

def create_valid_sequence(path: Path, video_id: str):
    seq_data = {
        "sequence_id": "seq123",
        "video_id": video_id,
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
                "label": "PHONE_USE"
            }
        ],
        "context": {
            "inside_vehicle": True,
            "outside_vehicle_person": False,
            "motorcycle_flag": False
        },
        "review_provenance": {
            "reviewer_type": "HUMAN",
            "status": "HUMAN_APPROVED",
            "reviewer_id": "rev1",
            "reviewed_at": "2026-09-04T12:05:00Z",
            "evidence_hash": "hash123",
            "annotation_version": "v1.0",
            "adjudication_status": "FINAL"
        }
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(seq_data, f)

def test_adapter_freezer_integration(tmp_path: Path):
    video_id = "test_vid_123"
    
    lock_path = tmp_path / "external_lock.json"
    create_valid_external_lock(lock_path, video_id)
    
    seq_path = tmp_path / "seq.json"
    create_valid_sequence(seq_path, video_id)
    
    schema = load_schema("datasets/schemas/v2_event_sequence_annotation.schema.json")
    
    csv_path = tmp_path / "events.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "video_id", "event_id", "event_type", "start_seconds", "end_seconds",
            "vehicle_id", "cabin_id", "occupant_role", "inside_vehicle",
            "outside_vehicle_person", "motorcycle_flag", "label",
            "visibility", "conditions",
            "human_review_status", "reviewer_id", "reviewer_type", "reviewed_at",
            "adjudication_status", "notes"
        ])
        success = process_file(seq_path, writer, schema)
        
    assert success, "Adapter failed to process the sequence JSON"
    
    output_frozen = tmp_path / "frozen.json"
    frozen_result = freeze_event_ground_truth(csv_path, lock_path, output_frozen)
    
    assert frozen_result["status"] == "FROZEN_EVENT_GROUND_TRUTH"
    assert frozen_result["rows"] == 1
    assert "PHONE" in frozen_result["event_type_counts"]
