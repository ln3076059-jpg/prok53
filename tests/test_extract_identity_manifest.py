from __future__ import annotations

import csv
import json
from pathlib import Path
import jsonschema
import pytest

from training.extract_identity_manifest import (
    extract_identities_from_predictions_csv,
    extract_identity_manifest_from_tracks,
    freeze_identity_manifest,
    generate_annotation_skeleton,
    generate_annotation_skeletons_for_video,
)
from training.evaluate_events import evaluate_frozen
from training.common import sha256_file
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


def test_extract_identity_manifest_from_tracks_preserves_full_provenance(tmp_path: Path):
    tracks_file = tmp_path / "runtime_identity_tracks.jsonl"
    record1 = {
        "video_id": "vid-provenance-1",
        "video_sha256": "v" * 64,
        "fps": 25.0,
        "frame_count": 3000,
        "duration": 120.0,
        "vehicle_id": "video:vid-provenance-1:vehicle-track:1",
        "cabin_id": "video:vid-provenance-1:vehicle-track:1:cabin:0",
        "occupant_id": "video:vid-provenance-1:vehicle-track:1:cabin:0:occupant-track:10",
        "first_frame": 0,
        "last_frame": 2999,
        "tracking_evidence": {
            "observation_count": 2850,
            "first_seconds": 0.0,
            "last_seconds": 119.96,
            "average_confidence": 0.92,
            "assigned_role": "driver",
        },
    }
    record2 = {
        "video_id": "vid-provenance-1",
        "video_sha256": "v" * 64,
        "fps": 25.0,
        "frame_count": 3000,
        "duration": 120.0,
        "vehicle_id": "video:vid-provenance-1:vehicle-track:1",
        "cabin_id": "video:vid-provenance-1:vehicle-track:1:cabin:0",
        "occupant_id": "video:vid-provenance-1:vehicle-track:1:cabin:0:occupant-track:11",
        "first_frame": 50,
        "last_frame": 2800,
        "tracking_evidence": {
            "observation_count": 2500,
            "first_seconds": 2.0,
            "last_seconds": 112.0,
            "average_confidence": 0.89,
            "assigned_role": "front_passenger",
        },
    }
    with tracks_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record1) + "\n")
        f.write(json.dumps(record2) + "\n")

    manifest = extract_identity_manifest_from_tracks(tracks_file)

    assert manifest["manifest_version"] == "v2.0"
    assert manifest["source_type"] == "RUNTIME_IDENTITY_TRACKS"
    assert manifest["source_sha256"] == sha256_file(tracks_file)
    assert manifest["source_path"] == str(tracks_file.resolve())

    v_data = manifest["videos"]["vid-provenance-1"]
    assert v_data["fps"] == 25.0
    assert v_data["frame_count"] == 3000
    assert v_data["duration_seconds"] == 120.0
    assert v_data["video_sha256"] == "v" * 64
    assert len(v_data["occupants"]) == 2

    lock_file = tmp_path / "tracks_manifest_lock.json"
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lock = freeze_identity_manifest(manifest_file, lock_file)

    assert lock["status"] == "FROZEN_IDENTITY_MANIFEST"
    assert lock["source_type"] == "RUNTIME_IDENTITY_TRACKS"
    assert lock["source_sha256"] == sha256_file(tracks_file)


def test_zero_predictions_manifest_extraction_does_not_blind_evaluator(tmp_path: Path):
    tracks_file = tmp_path / "runtime_identity_tracks.jsonl"
    record = {
        "video_id": "vid-safe-driver",
        "video_sha256": "s" * 64,
        "fps": 30.0,
        "frame_count": 3600,
        "duration": 120.0,
        "vehicle_id": "video:vid-safe-driver:vehicle-track:1",
        "cabin_id": "video:vid-safe-driver:vehicle-track:1:cabin:0",
        "occupant_id": "video:vid-safe-driver:vehicle-track:1:cabin:0:occupant-track:1",
        "first_frame": 0,
        "last_frame": 3599,
        "tracking_evidence": {
            "observation_count": 3590,
            "first_seconds": 0.0,
            "last_seconds": 119.97,
            "average_confidence": 0.95,
            "assigned_role": "driver",
        },
    }
    with tracks_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(record) + "\n")

    manifest = extract_identity_manifest_from_tracks(tracks_file)
    assert len(manifest["proven_identities"]) == 1
    assert manifest["videos"]["vid-safe-driver"]["duration_seconds"] == 120.0

    manifest_file = tmp_path / "manifest_safe.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lock_file = tmp_path / "manifest_safe_lock.json"
    lock = freeze_identity_manifest(manifest_file, lock_file)

    external_lock_file = tmp_path / "external-lock-safe.json"
    external_lock_file.write_text(
        json.dumps(
            {
                "status": "FROZEN_EXTERNAL_TEST",
                "human_review_status": "ALL_APPROVED",
                "video_ids": ["vid-safe-driver"],
                "identity_manifest_sha256": lock["manifest_sha256"],
                "identity_manifest_lock_path": str(lock_file.resolve()),
                "require_identity_manifest": True,
            }
        ),
        encoding="utf-8",
    )

    truth_csv = tmp_path / "event_truth_missed.csv"
    with truth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "video_id": "vid-safe-driver",
                "event_id": "evt-missed-1",
                "event_type": "PHONE",
                "start_seconds": "15.0",
                "end_seconds": "20.0",
                "occupant_id": "video:vid-safe-driver:vehicle-track:1:cabin:0:occupant-track:1",
                "vehicle_id": "video:vid-safe-driver:vehicle-track:1",
                "cabin_id": "video:vid-safe-driver:vehicle-track:1:cabin:0",
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
                "notes": "missed phone use event by model",
            }
        )

    frozen_event = freeze_event_ground_truth(
        truth_csv,
        external_lock_file,
        tmp_path / "frozen_event_missed.json",
    )
    assert frozen_event["identity_manifest_sha256"] == lock["manifest_sha256"]


def test_freeze_identity_manifest_cryptographic_verification(tmp_path: Path):
    tracks_file = tmp_path / "tracks.jsonl"
    tracks_file.write_text(
        json.dumps(
            {
                "video_id": "v1",
                "vehicle_id": "video:v1:vehicle-track:1",
                "cabin_id": "video:v1:vehicle-track:1:cabin:0",
                "occupant_id": "video:v1:vehicle-track:1:cabin:0:occupant-track:1",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    manifest = extract_identity_manifest_from_tracks(tracks_file)

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")

    # 1. Missing source path
    manifest_bad_path = {**manifest, "source_path": str(tmp_path / "non_existent.jsonl")}
    m_path = tmp_path / "manifest_bad_path.json"
    m_path.write_text(json.dumps(manifest_bad_path), encoding="utf-8")
    with pytest.raises(ValueError, match="source_path does not exist on disk"):
        freeze_identity_manifest(m_path, tmp_path / "lock1.json")

    # 2. Tampered source file / SHA mismatch
    manifest_tampered = {**manifest, "source_sha256": "0" * 64}
    m_tampered = tmp_path / "manifest_tampered.json"
    m_tampered.write_text(json.dumps(manifest_tampered), encoding="utf-8")
    with pytest.raises(ValueError, match="source_sha256 mismatch"):
        freeze_identity_manifest(m_tampered, tmp_path / "lock2.json")

    # 3. Invalid source_type
    manifest_bad_type = {**manifest, "source_type": "UNTRUSTED_INFERENCE"}
    m_type = tmp_path / "manifest_bad_type.json"
    m_type.write_text(json.dumps(manifest_bad_type), encoding="utf-8")
    with pytest.raises(ValueError, match="unrecognized identity manifest source_type"):
        freeze_identity_manifest(m_type, tmp_path / "lock3.json")


def test_multi_cabin_video_generates_isolated_skeletons(tmp_path: Path):
    tracks_file = tmp_path / "multi_cabin_tracks.jsonl"
    r1 = {
        "video_id": "vid-multi",
        "fps": 30.0,
        "frame_count": 900,
        "duration": 30.0,
        "vehicle_id": "video:vid-multi:vehicle-track:1",
        "cabin_id": "video:vid-multi:vehicle-track:1:cabin:0",
        "occupant_id": "video:vid-multi:vehicle-track:1:cabin:0:occupant-track:1",
        "first_frame": 0,
        "last_frame": 899,
        "tracking_evidence": {"observation_count": 890, "assigned_role": "driver"},
    }
    r2 = {
        "video_id": "vid-multi",
        "fps": 30.0,
        "frame_count": 900,
        "duration": 30.0,
        "vehicle_id": "video:vid-multi:vehicle-track:2",
        "cabin_id": "video:vid-multi:vehicle-track:2:cabin:0",
        "occupant_id": "video:vid-multi:vehicle-track:2:cabin:0:occupant-track:2",
        "first_frame": 0,
        "last_frame": 899,
        "tracking_evidence": {"observation_count": 890, "assigned_role": "driver"},
    }
    with tracks_file.open("w", encoding="utf-8") as f:
        f.write(json.dumps(r1) + "\n")
        f.write(json.dumps(r2) + "\n")

    manifest = extract_identity_manifest_from_tracks(tracks_file)
    manifest_file = tmp_path / "manifest_multi.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lock_file = tmp_path / "lock_multi.json"
    lock = freeze_identity_manifest(manifest_file, lock_file)

    skeletons = generate_annotation_skeletons_for_video(
        manifest, "vid-multi", manifest_lock=lock
    )
    assert len(skeletons) == 2

    schema_path = Path("datasets/schemas/v2_event_sequence_annotation.schema.json")
    schema = load_schema(schema_path)

    sk1, sk2 = skeletons[0], skeletons[1]
    assert sk1["cabin_id"] == "video:vid-multi:vehicle-track:1:cabin:0"
    assert len(sk1["occupants"]) == 1
    assert sk1["occupants"][0]["occupant_id"] == "video:vid-multi:vehicle-track:1:cabin:0:occupant-track:1"
    assert sk1["review_provenance"]["identity_manifest_sha256"] == lock["manifest_sha256"]

    assert sk2["cabin_id"] == "video:vid-multi:vehicle-track:2:cabin:0"
    assert len(sk2["occupants"]) == 1
    assert sk2["occupants"][0]["occupant_id"] == "video:vid-multi:vehicle-track:2:cabin:0:occupant-track:2"
    assert sk2["review_provenance"]["identity_manifest_sha256"] == lock["manifest_sha256"]

    assert validate_annotation(sk1, schema, allow_proposal=True) == []
    assert validate_annotation(sk2, schema, allow_proposal=True) == []


def test_proposal_skeleton_no_fabrication_and_binds_lock_sha(tmp_path: Path):
    tracks_file = tmp_path / "tracks.jsonl"
    r = {
        "video_id": "vid-proposal-test",
        "fps": 30.0,
        "frame_count": 300,
        "duration": 10.0,
        "vehicle_id": "video:vid-proposal-test:vehicle-track:1",
        "cabin_id": "video:vid-proposal-test:vehicle-track:1:cabin:0",
        "occupant_id": "video:vid-proposal-test:vehicle-track:1:cabin:0:occupant-track:1",
        "first_frame": 0,
        "last_frame": 299,
        "tracking_evidence": {"observation_count": 290, "assigned_role": "driver"},
    }
    tracks_file.write_text(json.dumps(r) + "\n", encoding="utf-8")
    manifest = extract_identity_manifest_from_tracks(tracks_file)
    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest), encoding="utf-8")
    lock = freeze_identity_manifest(manifest_file, tmp_path / "lock.json")

    skeleton = generate_annotation_skeleton(
        manifest, "vid-proposal-test", manifest_lock=lock
    )

    # Asserts no fabrication of safety-context booleans
    assert skeleton["context"]["inside_vehicle"] is None
    assert skeleton["context"]["outside_vehicle_person"] is None
    assert skeleton["context"]["motorcycle_flag"] is None

    for interval in skeleton["context_intervals"]:
        assert interval["inside_vehicle"] is None
        assert interval["outside_vehicle_person"] is None
        assert interval["motorcycle_flag"] is None
        assert interval["phone_state"] == "UNKNOWN"
        assert interval["seatbelt_state"] == "UNCERTAIN_OR_OCCLUDED"
        assert interval["visibility"] == "UNREVIEWED"
        assert interval["conditions"] == "UNREVIEWED"

    assert skeleton["review_provenance"]["identity_manifest_sha256"] == lock["manifest_sha256"]
    schema = load_schema(Path("datasets/schemas/v2_event_sequence_annotation.schema.json"))
    assert validate_annotation(skeleton, schema, allow_proposal=True) == []

