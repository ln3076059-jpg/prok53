import json
import csv
from pathlib import Path
import tempfile
import pytest

from training.validate_event_sequence_annotations import load_schema
from training.build_event_truth_from_sequences import process_file
from training.freeze_context_ground_truth import freeze_context_ground_truth
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
        "vehicle_id": "vehicle-1",
        "cabin_id": "cabin-1",
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
        "context_intervals": [
            {
                "context_id": "ctx-before",
                "occupant_id": "occ1",
                "start_frame": 0,
                "end_frame": 30,
                "inside_vehicle": True,
                "outside_vehicle_person": False,
                "motorcycle_flag": False,
                "phone_state": "NO_PHONE",
                "seatbelt_state": "FASTENED",
                "visibility": "clear",
                "conditions": "daylight",
            },
            {
                "context_id": "ctx-event",
                "occupant_id": "occ1",
                "start_frame": 30,
                "end_frame": 61,
                "inside_vehicle": True,
                "outside_vehicle_person": False,
                "motorcycle_flag": False,
                "phone_state": "PHONE_USE",
                "seatbelt_state": "FASTENED",
                "visibility": "clear",
                "conditions": "daylight",
            },
            {
                "context_id": "ctx-after",
                "occupant_id": "occ1",
                "start_frame": 61,
                "end_frame": 300,
                "inside_vehicle": True,
                "outside_vehicle_person": False,
                "motorcycle_flag": False,
                "phone_state": "NO_PHONE",
                "seatbelt_state": "FASTENED",
                "visibility": "clear",
                "conditions": "daylight",
            },
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
            "occupant_id", "vehicle_id", "cabin_id", "occupant_role", "inside_vehicle",
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

def test_adapter_freezer_cli_integration(tmp_path: Path):
    import subprocess
    import sys
    
    video_id = "test_vid_cli"
    
    lock_path = tmp_path / "external_lock.json"
    create_valid_external_lock(lock_path, video_id)
    
    seq_dir = tmp_path / "sequences"
    seq_dir.mkdir()
    seq_path = seq_dir / "seq1.json"
    create_valid_sequence(seq_path, video_id)
    
    csv_path = tmp_path / "events.csv"
    context_path = tmp_path / "context.csv"
    
    # Run the adapter via CLI
    result = subprocess.run([
        sys.executable, "-m", "training.build_event_truth_from_sequences",
        str(seq_dir), str(csv_path),
        "--context-output", str(context_path)
    ], capture_output=True, text=True)
    
    assert result.returncode == 0, f"Adapter CLI failed: {result.stderr}"
    assert csv_path.exists(), "Adapter CLI did not produce CSV"
    assert context_path.exists(), "Adapter CLI did not produce context CSV"
    
    # Freeze the output
    output_frozen = tmp_path / "frozen.json"
    frozen_result = freeze_event_ground_truth(csv_path, lock_path, output_frozen)
    frozen_context = freeze_context_ground_truth(
        context_path, lock_path, tmp_path / "context-frozen.json"
    )
    
    assert frozen_result["status"] == "FROZEN_EVENT_GROUND_TRUTH"
    assert frozen_result["rows"] == 1
    assert "PHONE" in frozen_result["event_type_counts"]
    assert frozen_context["coverage_status"] == "FULL_TIMELINE_NO_GAPS_OR_OVERLAPS"
