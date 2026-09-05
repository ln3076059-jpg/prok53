from __future__ import annotations

import csv
import json
from pathlib import Path
import jsonschema
import pytest

from training.extract_identity_manifest import (
    extract_identities_from_predictions_csv,
    freeze_identity_manifest,
    generate_annotation_skeleton,
)
from training.freeze_event_ground_truth import REQUIRED_COLUMNS, freeze_event_ground_truth
from training.freeze_context_ground_truth import (
    REQUIRED_COLUMNS as CONTEXT_REQUIRED_COLUMNS,
    freeze_context_ground_truth,
)
from training.identity_contract import validate_identity_contract
from training.validate_event_sequence_annotations import load_schema, validate_annotation


def _make_sample_predictions_csv(csv_path: Path) -> None:
    fields = [
        "video_id",
        "event_type",
        "occupant_id",
        "track_id",
        "occupant_role",
        "vehicle_id",
        "cabin_id",
        "start_seconds",
        "end_seconds",
        "label",
        "visibility",
        "outside_vehicle_person",
        "motorcycle_flag",
        "observation_count",
    ]
    rows = [
        {
            "video_id": "vid-run-1",
            "event_type": "PHONE",
            "occupant_id": "video:vid-run-1:vehicle-track:1:cabin:0:occupant-track:42",
            "track_id": "42",
            "occupant_role": "driver",
            "vehicle_id": "video:vid-run-1:vehicle-track:1",
            "cabin_id": "video:vid-run-1:vehicle-track:1:cabin:0",
            "start_seconds": "1.0",
            "end_seconds": "3.5",
            "label": "PHONE_USE",
            "visibility": "clear",
            "outside_vehicle_person": "false",
            "motorcycle_flag": "false",
            "observation_count": "10",
        },
        {
            "video_id": "vid-run-1",
            "event_type": "PHONE",
            "occupant_id": "video:vid-run-1:vehicle-track:1:cabin:0:occupant-track:43",
            "track_id": "43",
            "occupant_role": "front_passenger",
            "vehicle_id": "video:vid-run-1:vehicle-track:1",
            "cabin_id": "video:vid-run-1:vehicle-track:1:cabin:0",
            "start_seconds": "2.0",
            "end_seconds": "4.0",
            "label": "PHONE_USE",
            "visibility": "clear",
            "outside_vehicle_person": "false",
            "motorcycle_flag": "false",
            "observation_count": "8",
        },
    ]
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def test_extract_identities_from_predictions_csv(tmp_path: Path):
    pred_csv = tmp_path / "predictions.csv"
    _make_sample_predictions_csv(pred_csv)

    manifest = extract_identities_from_predictions_csv(pred_csv)
    assert manifest["manifest_version"] == "v2.0"
    assert manifest["source_type"] == "RUNTIME_PREDICTIONS_CSV"
    assert len(manifest["source_sha256"]) == 64
    assert "vid-run-1" in manifest["videos"]

    video_info = manifest["videos"]["vid-run-1"]
    assert video_info["vehicle_ids"] == ["video:vid-run-1:vehicle-track:1"]
    assert video_info["cabin_ids"] == ["video:vid-run-1:vehicle-track:1:cabin:0"]
    assert len(video_info["occupants"]) == 2

    # Check proven identities list
    proven = manifest["proven_identities"]
    assert len(proven) == 2
    occ_ids = {p["occupant_id"] for p in proven}
    assert "video:vid-run-1:vehicle-track:1:cabin:0:occupant-track:42" in occ_ids
    assert "video:vid-run-1:vehicle-track:1:cabin:0:occupant-track:43" in occ_ids


def test_freeze_identity_manifest(tmp_path: Path):
    pred_csv = tmp_path / "predictions.csv"
    _make_sample_predictions_csv(pred_csv)
    manifest = extract_identities_from_predictions_csv(pred_csv)

    manifest_file = tmp_path / "identity_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    lock_file = tmp_path / "identity_manifest_lock.json"
    lock_data = freeze_identity_manifest(manifest_file, lock_file)

    assert lock_data["status"] == "FROZEN_IDENTITY_MANIFEST"
    assert len(lock_data["manifest_sha256"]) == 64
    assert lock_data["identity_count"] == 2
    assert lock_file.exists()

    # Refuse overwrite
    with pytest.raises(FileExistsError):
        freeze_identity_manifest(manifest_file, lock_file)


def test_generate_annotation_skeleton_no_fabrication_and_semantic_valid(tmp_path: Path):
    pred_csv = tmp_path / "predictions.csv"
    _make_sample_predictions_csv(pred_csv)
    manifest = extract_identities_from_predictions_csv(pred_csv)

    skeleton = generate_annotation_skeleton(manifest, "vid-run-1")

    # 1. Structural and temporal correctness
    assert skeleton["video_id"] == "vid-run-1"
    assert skeleton["vehicle_id"] == "video:vid-run-1:vehicle-track:1"
    assert skeleton["cabin_id"] == "video:vid-run-1:vehicle-track:1:cabin:0"
    assert skeleton["start_time"] < skeleton["end_time"]

    # 2. No fabricated business ground truth
    assert skeleton["events"] == []
    for occ in skeleton["occupants"]:
        assert occ["role"] == "unknown"
        assert occ["reviewer_confirmed_role"] is False

    for interval in skeleton["context_intervals"]:
        assert interval["phone_state"] == "UNKNOWN"
        assert interval["seatbelt_state"] == "UNCERTAIN_OR_OCCLUDED"
        assert interval["visibility"] == "UNREVIEWED"
        assert interval["conditions"] == "UNREVIEWED"
        assert interval["notes"] == "SKELETON_UNREVIEWED_IDENTITY_PROVENANCE"

    assert skeleton["review_provenance"]["reviewer_type"] == "AI"
    assert skeleton["review_provenance"]["status"] == "AI_REVIEWED_PROPOSAL"

    # 3. Validates against schema
    schema_path = Path("datasets/schemas/v2_event_sequence_annotation.schema.json")
    schema = load_schema(schema_path)
    jsonschema.validate(instance=skeleton, schema=schema)

    # 4. Validates against semantic validator with allow_proposal=True
    semantic_errors = validate_annotation(skeleton, schema, allow_proposal=True)
    assert semantic_errors == []


def test_freezer_rejects_unproven_identities(tmp_path: Path):
    # Setup runtime manifest with occupant 42 only
    pred_csv = tmp_path / "predictions.csv"
    _make_sample_predictions_csv(pred_csv)
    manifest = extract_identities_from_predictions_csv(pred_csv)
    # Restrict to only track 42
    manifest["videos"]["vid-run-1"]["occupants"] = [
        manifest["videos"]["vid-run-1"]["occupants"][0]
    ]
    manifest["proven_identities"] = [manifest["proven_identities"][0]]

    manifest_file = tmp_path / "identity_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lock_file = tmp_path / "identity_manifest_lock.json"
    lock_data = freeze_identity_manifest(manifest_file, lock_file)

    external_lock_file = tmp_path / "external-lock.json"
    external_lock_file.write_text(
        json.dumps(
            {
                "status": "FROZEN_EXTERNAL_TEST",
                "human_review_status": "ALL_APPROVED",
                "video_ids": ["vid-run-1"],
            }
        ),
        encoding="utf-8",
    )

    # 1. Event truth with unproven occupant 99 (not in manifest) -> REJECTED
    truth_csv = tmp_path / "event_truth_unproven.csv"
    with truth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "video_id": "vid-run-1",
                "event_id": "evt-1",
                "event_type": "PHONE",
                "start_seconds": "1.0",
                "end_seconds": "2.0",
                "occupant_id": "video:vid-run-1:vehicle-track:1:cabin:0:occupant-track:99",
                "vehicle_id": "video:vid-run-1:vehicle-track:1",
                "cabin_id": "video:vid-run-1:vehicle-track:1:cabin:0",
                "inside_vehicle": "true",
                "outside_vehicle_person": "false",
                "motorcycle_flag": "false",
                "label": "PHONE_USE",
                "occupant_role": "driver",
                "visibility": "clear",
                "conditions": "daylight",
                "human_review_status": "APPROVED",
                "reviewer_id": "rev-1",
                "reviewer_type": "HUMAN",
                "reviewed_at": "2026-09-05T00:00:00Z",
                "adjudication_status": "FINAL",
                "notes": "unproven occupant",
            }
        )

    with pytest.raises(ValueError, match="is not in the frozen identity manifest"):
        freeze_event_ground_truth(
            truth_csv,
            external_lock_file,
            tmp_path / "frozen_event.json",
            identity_manifest_lock_path=lock_file,
        )

    # 2. Context truth with unproven occupant 99 -> REJECTED
    context_csv = tmp_path / "context_truth_unproven.csv"
    with context_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(CONTEXT_REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "video_id": "vid-run-1",
                "context_id": "ctx-1",
                "occupant_id": "video:vid-run-1:vehicle-track:1:cabin:0:occupant-track:99",
                "vehicle_id": "video:vid-run-1:vehicle-track:1",
                "cabin_id": "video:vid-run-1:vehicle-track:1:cabin:0",
                "occupant_role": "driver",
                "start_seconds": "0",
                "end_seconds": "60",
                "timeline_end_seconds": "60",
                "inside_vehicle": "true",
                "outside_vehicle_person": "false",
                "motorcycle_flag": "false",
                "phone_state": "NO_PHONE",
                "seatbelt_state": "FASTENED",
                "visibility": "clear",
                "conditions": "daylight",
                "human_review_status": "APPROVED",
                "reviewer_id": "rev-1",
                "reviewer_type": "HUMAN",
                "reviewed_at": "2026-09-05T00:00:00Z",
                "adjudication_status": "FINAL",
                "notes": "unproven occupant context",
            }
        )

    with pytest.raises(ValueError, match="is not in the frozen identity manifest"):
        freeze_context_ground_truth(
            context_csv,
            external_lock_file,
            tmp_path / "frozen_context.json",
            identity_manifest_lock_path=lock_file,
        )

    # 3. Event truth with PROVEN occupant 42 -> SUCCESS and binds manifest SHA
    truth_csv_proven = tmp_path / "event_truth_proven.csv"
    with truth_csv_proven.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "video_id": "vid-run-1",
                "event_id": "evt-1",
                "event_type": "PHONE",
                "start_seconds": "1.0",
                "end_seconds": "2.0",
                "occupant_id": "video:vid-run-1:vehicle-track:1:cabin:0:occupant-track:42",
                "vehicle_id": "video:vid-run-1:vehicle-track:1",
                "cabin_id": "video:vid-run-1:vehicle-track:1:cabin:0",
                "inside_vehicle": "true",
                "outside_vehicle_person": "false",
                "motorcycle_flag": "false",
                "label": "PHONE_USE",
                "occupant_role": "driver",
                "visibility": "clear",
                "conditions": "daylight",
                "human_review_status": "APPROVED",
                "reviewer_id": "rev-1",
                "reviewer_type": "HUMAN",
                "reviewed_at": "2026-09-05T00:00:00Z",
                "adjudication_status": "FINAL",
                "notes": "proven occupant",
            }
        )

    frozen_event = freeze_event_ground_truth(
        truth_csv_proven,
        external_lock_file,
        tmp_path / "frozen_event_proven.json",
        identity_manifest_lock_path=lock_file,
    )
    assert frozen_event["identity_manifest_sha256"] == lock_data["manifest_sha256"]
