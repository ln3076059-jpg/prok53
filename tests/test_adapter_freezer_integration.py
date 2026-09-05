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
    vehicle_id = f"video:{video_id}:vehicle-track:1"
    cabin_id = f"{vehicle_id}:cabin:1"
    occupant_id = f"{cabin_id}:occupant-track:1"
    manifest_lock_path = path.parent / f"identity_manifest_{video_id}.json"
    manifest_sha = "a" * 64
    manifest_lock = {
        "status": "FROZEN_IDENTITY_MANIFEST",
        "manifest_sha256": manifest_sha,
        "manifest_file": "manifest.json",
        "source_type": "RUNTIME_IDENTITY_TRACKS",
        "source_path": f"memory:tracks:{video_id}",
        "source_sha256": "b" * 64,
        "eligible_for_frozen_event_evaluation": True,
        "evaluation_scope": "FULL_SYSTEM_EVENT_EVALUATION",
        "video_ids": [video_id],
        "videos": {video_id: {"sha256": "v" * 64, "fps": 30.0, "frame_count": 1800, "duration_seconds": 60.0}},
        "proven_identities": [
            {
                "video_id": video_id,
                "vehicle_id": vehicle_id,
                "cabin_id": cabin_id,
                "occupant_id": occupant_id,
            }
        ],
        "identity_count": 1,
        "locked_at": "2026-09-05T00:00:00Z",
    }
    with open(manifest_lock_path, "w", encoding="utf-8") as f:
        json.dump(manifest_lock, f)

    lock_data = {
        "status": "FROZEN_EXTERNAL_TEST",
        "human_review_status": "ALL_APPROVED",
        "video_ids": [video_id],
        "identity_manifest_sha256": manifest_sha,
        "identity_manifest_lock_path": str(manifest_lock_path.resolve()),
        "require_identity_manifest": True,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(lock_data, f)

def create_valid_sequence(path: Path, video_id: str):
    vehicle_id = f"video:{video_id}:vehicle-track:1"
    cabin_id = f"{vehicle_id}:cabin:1"
    occupant_id = f"{cabin_id}:occupant-track:1"
    seq_data = {
        "sequence_id": "seq123",
        "video_id": video_id,
        "vehicle_id": vehicle_id,
        "cabin_id": cabin_id,
        "source_id": "src1",
        "fps": 30.0,
        "frame_count": 300,
        "start_time": "2026-09-04T12:00:00Z",
        "end_time": "2026-09-04T12:00:10Z",
        "occupants": [
            {"occupant_id": occupant_id, "role": "driver", "role_confidence": 0.9}
        ],
        "events": [
            {
                "event_type": "PHONE",
                "occupant_id": occupant_id,
                "start_frame": 30,
                "end_frame": 60,
                "label": "PHONE_USE"
            }
        ],
        "context_intervals": [
            {
                "context_id": "ctx-before",
                "occupant_id": occupant_id,
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
                "occupant_id": occupant_id,
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
                "occupant_id": occupant_id,
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
            "identity_manifest_sha256": "a" * 64,
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
    
    from training.build_event_truth_from_sequences import EVENT_FIELDNAMES
    csv_path = tmp_path / "events.csv"
    with open(csv_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(EVENT_FIELDNAMES)
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
