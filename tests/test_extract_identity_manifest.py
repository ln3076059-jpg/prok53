from __future__ import annotations

import csv
import json
from pathlib import Path
import jsonschema
import pytest

from training.extract_identity_manifest import (
    create_identity_roster,
    extract_identities_from_predictions_csv,
    extract_identity_manifest_from_annotations,
    extract_identity_manifest_from_roster,
    extract_identity_manifest_from_tracks,
    freeze_identity_adjudication,
    freeze_identity_manifest,
    generate_annotation_skeleton,
    generate_annotation_skeletons_for_video,
)
from training.evaluate_events import evaluate, evaluate_frozen, verify_evaluation_integrity
from training.freeze_external_test import freeze_external_test
from training.build_event_truth_from_sequences import (
    process_file,
    EVENT_FIELDNAMES,
    CONTEXT_FIELDNAMES,
)
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
    manifest["source_type"] = "RUNTIME_IDENTITY_TRACKS"
    manifest["eligible_for_frozen_event_evaluation"] = True
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
                "identity_manifest_sha256": lock_data["manifest_sha256"],
                "identity_manifest_lock_path": str(lock_file.resolve()),
                "require_identity_manifest": True,
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
                "identity_manifest_sha256": lock_data["manifest_sha256"],
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
                "identity_manifest_sha256": lock_data["manifest_sha256"],
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
                "identity_manifest_sha256": lock_data["manifest_sha256"],
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
                "identity_manifest_sha256": lock["manifest_sha256"],
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


def test_extract_identity_manifest_from_annotations(tmp_path: Path):
    ann_path = tmp_path / "seq.json"
    ann_data = {
        "sequence_id": "seq-ind-1",
        "video_id": "vid-annotated",
        "vehicle_id": "video:vid-annotated:vehicle-track:1",
        "cabin_id": "video:vid-annotated:vehicle-track:1:cabin:0",
        "fps": 25.0,
        "frame_count": 2500,
        "video_sha256": "v" * 64,
        "occupants": [
            {
                "occupant_id": "video:vid-annotated:vehicle-track:1:cabin:0:occupant-track:1",
                "role": "driver",
            },
            {
                "occupant_id": "video:vid-annotated:vehicle-track:1:cabin:0:occupant-track:2",
                "role": "front_passenger",
            },
        ],
        "events": [],
        "context_intervals": [],
    }
    ann_path.write_text(json.dumps(ann_data, indent=2), encoding="utf-8")

    manifest = extract_identity_manifest_from_annotations([ann_path])

    assert manifest["manifest_version"] == "v2.0"
    assert manifest["source_type"] == "INDEPENDENT_GROUND_TRUTH_ANNOTATIONS"
    assert manifest["evaluation_scope"] == "LEGACY_ANNOTATION_EXTRACTED_SCOPE"
    assert manifest["eligible_for_frozen_event_evaluation"] is False
    assert len(manifest["proven_identities"]) == 2
    assert manifest["videos"]["vid-annotated"]["duration_seconds"] == 100.0

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lock_file = tmp_path / "manifest_lock.json"
    lock = freeze_identity_manifest(manifest_file, lock_file)

    assert lock["status"] == "FROZEN_IDENTITY_MANIFEST"
    assert lock["eligible_for_frozen_event_evaluation"] is False
    assert lock["evaluation_scope"] == "LEGACY_ANNOTATION_EXTRACTED_SCOPE"
    assert lock["videos"]["vid-annotated"]["sha256"] == "v" * 64

    # Missing video SHA in annotation must fail-closed (no fake fallback!)
    ann_no_sha = dict(ann_data)
    del ann_no_sha["video_sha256"]
    ann_no_sha_path = tmp_path / "ann_no_sha.json"
    ann_no_sha_path.write_text(json.dumps(ann_no_sha, indent=2), encoding="utf-8")
    with pytest.raises(ValueError, match="missing valid 64-character video_sha256"):
        extract_identity_manifest_from_annotations([ann_no_sha_path])


def test_freezer_and_evaluator_reject_legacy_manifest_locks(tmp_path: Path):
    pred_csv = tmp_path / "preds.csv"
    _make_sample_predictions_csv(pred_csv)
    manifest = extract_identities_from_predictions_csv(pred_csv)
    assert manifest["eligible_for_frozen_event_evaluation"] is False

    manifest_file = tmp_path / "legacy_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lock_file = tmp_path / "legacy_manifest_lock.json"
    lock = freeze_identity_manifest(manifest_file, lock_file)

    assert lock["eligible_for_frozen_event_evaluation"] is False

    ext_lock = tmp_path / "ext_lock.json"
    ext_lock.write_text(
        json.dumps(
            {
                "status": "FROZEN_EXTERNAL_TEST",
                "human_review_status": "ALL_APPROVED",
                "video_ids": ["vid-run-1"],
                "identity_manifest_sha256": lock["manifest_sha256"],
                "identity_manifest_lock_path": str(lock_file.resolve()),
            }
        ),
        encoding="utf-8",
    )

    truth_csv = tmp_path / "truth.csv"
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
                "notes": "test",
                "identity_manifest_sha256": lock["manifest_sha256"],
            }
        )

    with pytest.raises(ValueError, match="is not eligible for frozen event evaluation"):
        freeze_event_ground_truth(truth_csv, ext_lock, tmp_path / "frozen_evt.json", lock_file)


def test_gt_only_occupant_counts_as_false_negative_without_detector_dependency():
    truth_rows = [
        {
            "video_id": "vid-gt-miss",
            "event_id": "evt-driver-phone",
            "event_type": "PHONE",
            "start_seconds": "5.0",
            "end_seconds": "10.0",
            "occupant_id": "video:vid-gt-miss:vehicle-track:1:cabin:0:occupant-track:1",
            "vehicle_id": "video:vid-gt-miss:vehicle-track:1",
            "cabin_id": "video:vid-gt-miss:vehicle-track:1:cabin:0",
            "occupant_role": "driver",
            "inside_vehicle": "true",
            "outside_vehicle_person": "false",
            "motorcycle_flag": "false",
            "label": "PHONE_USE",
        }
    ]
    # Model runtime occupant detector missed the occupant completely (0 predictions)
    prediction_rows = []

    report = evaluate(
        truth_rows,
        prediction_rows,
        video_minutes=1.0,
    )

    phone_metrics = report["event_types"]["PHONE"]
    assert phone_metrics["true_positives"] == 0
    assert phone_metrics["missed_events"] == 1
    assert phone_metrics["recall"] == 0.0
    assert report["identity_adjudication"]["total_gt_occupants"] == 1
    assert report["identity_adjudication"]["tracked_occupants"] == 0
    assert report["identity_adjudication"]["untracked_gt_occupants"] == [
        "video:vid-gt-miss:vehicle-track:1:cabin:0:occupant-track:1"
    ]


def test_identity_adjudication_maps_runtime_track_to_gt_occupant():
    truth_rows = [
        {
            "video_id": "vid-1",
            "event_id": "evt-1",
            "event_type": "PHONE",
            "start_seconds": "5.0",
            "end_seconds": "10.0",
            "occupant_id": "video:vid-1:vehicle-track:1:cabin:0:occupant-track:1",
            "vehicle_id": "video:vid-1:vehicle-track:1",
            "cabin_id": "video:vid-1:vehicle-track:1:cabin:0",
            "occupant_role": "driver",
            "inside_vehicle": "true",
            "outside_vehicle_person": "false",
            "motorcycle_flag": "false",
            "label": "PHONE_USE",
        }
    ]
    # Model detected occupant as arbitrary runtime track 77
    prediction_rows = [
        {
            "video_id": "vid-1",
            "event_type": "PHONE",
            "start_seconds": "5.1",
            "end_seconds": "9.9",
            "occupant_id": "video:vid-1:vehicle-track:1:cabin:0:occupant-track:77",
            "vehicle_id": "video:vid-1:vehicle-track:1",
            "cabin_id": "video:vid-1:vehicle-track:1:cabin:0",
            "occupant_role": "driver",
            "inside_vehicle": "true",
            "outside_vehicle_person": "false",
            "motorcycle_flag": "false",
            "label": "PHONE_USE",
        }
    ]

    # Without mapping -> 0 matches, 1 missed event
    unmapped_report = evaluate(truth_rows, prediction_rows, video_minutes=1.0)
    assert unmapped_report["event_types"]["PHONE"]["true_positives"] == 0
    assert unmapped_report["event_types"]["PHONE"]["missed_events"] == 1

    # With adjudication mapping 77 -> 1
    mapping = {
        "video:vid-1:vehicle-track:1:cabin:0:occupant-track:77": "video:vid-1:vehicle-track:1:cabin:0:occupant-track:1"
    }
    mapped_report = evaluate(truth_rows, prediction_rows, video_minutes=1.0, identity_mapping=mapping)
    assert mapped_report["event_types"]["PHONE"]["true_positives"] == 1
    assert mapped_report["event_types"]["PHONE"]["missed_events"] == 0
    assert mapped_report["event_types"]["PHONE"]["recall"] == 1.0
    assert mapped_report["identity_adjudication"]["adjudication_applied"] is True


def test_freeze_external_test_enforces_video_sha_binding_with_manifest(tmp_path: Path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """schema_version: 1
dataset_role: EXTERNAL_TEST
require_identity_manifest: true
required_fields:
  - sample_id
  - dataset_role
  - source_id
  - camera_id
  - video_id
  - vehicle_id
  - person_id
  - video_path
  - sha256
  - annotation_path
  - annotation_sha256
  - conditions
  - human_review_status
  - reviewer_id
  - reviewer_type
  - reviewed_at
required_condition_coverage: [daylight]
disjoint_dimensions: [sha256, source_id, camera_id, video_id, vehicle_id, person_id]
minimum_independent_groups: {source_id: 1, camera_id: 1, video_id: 1, vehicle_id: 1, person_id: 1}
""",
        encoding="utf-8",
    )

    vid_file = tmp_path / "v1.mp4"
    vid_file.write_bytes(b"sample video bytes")
    v_sha = sha256_file(vid_file)

    ann_file = tmp_path / "ann.json"
    ann_file.write_text("[]", encoding="utf-8")
    a_sha = sha256_file(ann_file)

    manifest_record = {
        "sample_id": "s1",
        "dataset_role": "EXTERNAL_TEST",
        "source_id": "src1",
        "camera_id": "cam1",
        "video_id": "vid-sha-bind",
        "vehicle_id": "veh1",
        "person_id": "p1",
        "video_path": str(vid_file.resolve()),
        "sha256": v_sha,
        "annotation_path": str(ann_file.resolve()),
        "annotation_sha256": a_sha,
        "conditions": ["daylight"],
        "human_review_status": "APPROVED",
        "reviewer_id": "rev1",
        "reviewer_type": "HUMAN",
        "reviewed_at": "2026-09-05T00:00:00Z",
    }
    ext_manifest_path = tmp_path / "ext_manifest.jsonl"
    ext_manifest_path.write_text(json.dumps(manifest_record) + "\n", encoding="utf-8")

    dev_manifest_path = tmp_path / "dev_manifest.jsonl"
    dev_manifest_path.write_text("", encoding="utf-8")

    # Identity manifest lock with MISMATCHED video SHA (64 characters)
    mismatched_lock = {
        "status": "FROZEN_IDENTITY_MANIFEST",
        "manifest_sha256": "m" * 64,
        "manifest_file": "man.json",
        "source_type": "RUNTIME_IDENTITY_TRACKS",
        "source_path": str(vid_file.resolve()),
        "source_sha256": v_sha,
        "eligible_for_frozen_event_evaluation": True,
        "evaluation_scope": "FULL_SYSTEM_EVENT_EVALUATION",
        "video_ids": ["vid-sha-bind"],
        "videos": {
            "vid-sha-bind": {
                "sha256": "different_sha_from_tracking" + "0" * 37,
                "fps": 30.0,
                "frame_count": 300,
                "duration_seconds": 10.0,
            }
        },
        "proven_identities": [],
    }
    mismatched_lock_path = tmp_path / "mismatched_id_lock.json"
    mismatched_lock_path.write_text(json.dumps(mismatched_lock), encoding="utf-8")

    with pytest.raises(ValueError, match="video SHA in identity manifest .* does not match external test video SHA"):
        freeze_external_test(
            ext_manifest_path,
            dev_manifest_path,
            policy_path,
            tmp_path / "frozen_ext.json",
            identity_manifest_lock_path=mismatched_lock_path,
        )

    # Empty video SHA in identity manifest lock -> REJECTED fail-closed
    empty_sha_lock = dict(mismatched_lock)
    empty_sha_lock["videos"] = {
        "vid-sha-bind": {
            "sha256": "",
            "fps": 30.0,
            "frame_count": 300,
            "duration_seconds": 10.0,
        }
    }
    empty_sha_lock_path = tmp_path / "empty_sha_id_lock.json"
    empty_sha_lock_path.write_text(json.dumps(empty_sha_lock), encoding="utf-8")

    with pytest.raises(ValueError, match="is missing or not 64 characters"):
        freeze_external_test(
            ext_manifest_path,
            dev_manifest_path,
            policy_path,
            tmp_path / "frozen_ext_empty.json",
            identity_manifest_lock_path=empty_sha_lock_path,
        )


def test_annotation_manifest_sha_chain_of_custody(tmp_path: Path):
    truth_csv = tmp_path / "truth.csv"
    with truth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(
            {
                "video_id": "vid-chain",
                "event_id": "evt-chain-1",
                "event_type": "PHONE",
                "start_seconds": "1.0",
                "end_seconds": "2.0",
                "occupant_id": "video:vid-chain:vehicle-track:1:cabin:0:occupant-track:1",
                "vehicle_id": "video:vid-chain:vehicle-track:1",
                "cabin_id": "video:vid-chain:vehicle-track:1:cabin:0",
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
                "notes": "chain test",
                "identity_manifest_sha256": "1" * 64,  # Annotation says reviewed against manifest 1111...
            }
        )

    ext_lock = tmp_path / "ext_lock.json"
    ext_lock.write_text(
        json.dumps(
            {
                "status": "FROZEN_EXTERNAL_TEST",
                "human_review_status": "ALL_APPROVED",
                "video_ids": ["vid-chain"],
                "identity_manifest_sha256": "2" * 64,  # Lock is 2222...
            }
        ),
        encoding="utf-8",
    )

    manifest_lock = tmp_path / "id_lock.json"
    manifest_lock.write_text(
        json.dumps(
            {
                "status": "FROZEN_IDENTITY_MANIFEST",
                "manifest_sha256": "2" * 64,
                "manifest_file": "man.json",
                "source_type": "RUNTIME_IDENTITY_TRACKS",
                "source_path": "memory:dummy",
                "source_sha256": "s" * 64,
                "eligible_for_frozen_event_evaluation": True,
                "evaluation_scope": "FULL_SYSTEM_EVENT_EVALUATION",
                "video_ids": ["vid-chain"],
                "videos": {"vid-chain": {"sha256": "v" * 64, "fps": 30.0, "frame_count": 300, "duration_seconds": 10.0}},
                "proven_identities": [
                    {
                        "video_id": "vid-chain",
                        "vehicle_id": "video:vid-chain:vehicle-track:1",
                        "cabin_id": "video:vid-chain:vehicle-track:1:cabin:0",
                        "occupant_id": "video:vid-chain:vehicle-track:1:cabin:0:occupant-track:1",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    # Attempting to freeze annotation reviewed against manifest 1 against manifest lock 2 is rejected
    with pytest.raises(ValueError, match="annotation identity_manifest_sha256 .* does not match frozen manifest lock SHA"):
        freeze_event_ground_truth(
            truth_csv,
            ext_lock,
            tmp_path / "frozen_out.json",
            identity_manifest_lock_path=manifest_lock,
        )


def test_extract_identity_manifest_from_roster(tmp_path: Path):
    roster_file = tmp_path / "identity_roster.json"
    video_sha = "a" * 64
    roster_data = create_identity_roster(
        [
            {
                "video_id": "vid-roster-1",
                "video_sha256": video_sha,
                "fps": 30.0,
                "frame_count": 600,
                "vehicles": [
                    {
                        "vehicle_id": "video:vid-roster-1:vehicle-track:1",
                        "cabins": [
                            {
                                "cabin_id": "video:vid-roster-1:vehicle-track:1:cabin:0",
                                "occupants": [
                                    {
                                        "occupant_id": "video:vid-roster-1:vehicle-track:1:cabin:0:occupant-track:1",
                                        "role": "driver",
                                    },
                                    {
                                        "occupant_id": "video:vid-roster-1:vehicle-track:1:cabin:0:occupant-track:2",
                                        "role": "front_passenger",
                                    },
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
        output_path=roster_file,
    )

    manifest = extract_identity_manifest_from_roster(roster_file)
    assert manifest["manifest_version"] == "v2.0"
    assert manifest["source_type"] == "INDEPENDENT_IDENTITY_ROSTER"
    assert manifest["evaluation_scope"] == "FULL_SYSTEM_EVENT_EVALUATION"
    assert manifest["eligible_for_frozen_event_evaluation"] is True
    assert len(manifest["proven_identities"]) == 2
    assert manifest["videos"]["vid-roster-1"]["video_sha256"] == video_sha

    manifest_file = tmp_path / "roster_manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lock_file = tmp_path / "roster_manifest_lock.json"
    lock = freeze_identity_manifest(manifest_file, lock_file)

    assert lock["status"] == "FROZEN_IDENTITY_MANIFEST"
    assert lock["source_type"] == "INDEPENDENT_IDENTITY_ROSTER"
    assert lock["evaluation_scope"] == "FULL_SYSTEM_EVENT_EVALUATION"
    assert lock["eligible_for_frozen_event_evaluation"] is True
    assert len(lock["manifest_sha256"]) == 64
    assert lock["review_provenance"]["human_review_status"] == "APPROVED"
    assert lock["review_provenance"]["reviewer_type"] == "HUMAN"

    # Human review enforcement: AI reviewer must be rejected
    ai_roster = tmp_path / "ai_roster.json"
    create_identity_roster(
        [
            {
                "video_id": "vid-ai",
                "video_sha256": "a" * 64,
                "fps": 30.0,
                "frame_count": 100,
                "vehicles": [
                    {
                        "vehicle_id": "video:vid-ai:vehicle-track:1",
                        "cabins": [
                            {
                                "cabin_id": "video:vid-ai:vehicle-track:1:cabin:0",
                                "occupants": [{"occupant_id": "video:vid-ai:vehicle-track:1:cabin:0:occupant-track:1"}],
                            }
                        ],
                    }
                ],
            }
        ],
        output_path=ai_roster,
        reviewer_type="AI",
    )
    ai_man = extract_identity_manifest_from_roster(ai_roster)
    ai_man_path = tmp_path / "ai_man.json"
    ai_man_path.write_text(json.dumps(ai_man), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer_type == 'HUMAN'"):
        freeze_identity_manifest(ai_man_path, tmp_path / "ai_lock.json")

    # Human review enforcement: Unapproved status must be rejected
    unapp_roster = tmp_path / "unapp_roster.json"
    create_identity_roster(
        [
            {
                "video_id": "vid-unapp",
                "video_sha256": "b" * 64,
                "fps": 30.0,
                "frame_count": 100,
                "vehicles": [
                    {
                        "vehicle_id": "video:vid-unapp:vehicle-track:1",
                        "cabins": [
                            {
                                "cabin_id": "video:vid-unapp:vehicle-track:1:cabin:0",
                                "occupants": [{"occupant_id": "video:vid-unapp:vehicle-track:1:cabin:0:occupant-track:1"}],
                            }
                        ],
                    }
                ],
            }
        ],
        output_path=unapp_roster,
        human_review_status="PENDING",
    )
    unapp_man = extract_identity_manifest_from_roster(unapp_roster)
    unapp_man_path = tmp_path / "unapp_man.json"
    unapp_man_path.write_text(json.dumps(unapp_man), encoding="utf-8")
    with pytest.raises(ValueError, match="human_review_status == 'APPROVED'"):
        freeze_identity_manifest(unapp_man_path, tmp_path / "unapp_lock.json")

    # Invalid video SHA in roster -> REJECTED
    invalid_roster = tmp_path / "invalid_roster.json"
    create_identity_roster(
        [
            {
                "video_id": "vid-bad",
                "video_sha256": "too_short",
                "fps": 30.0,
                "frame_count": 100,
                "vehicles": [],
            }
        ],
        output_path=invalid_roster,
    )
    with pytest.raises(ValueError, match="missing valid 64-character video_sha256"):
        extract_identity_manifest_from_roster(invalid_roster)


def test_roster_to_annotation_skeleton_pipeline_eliminates_circular_sha(tmp_path: Path):
    # Step 1: Create independent roster and freeze identity manifest
    roster_file = tmp_path / "roster.json"
    v_sha = "f" * 64
    create_identity_roster(
        [
            {
                "video_id": "vid-streamline",
                "video_sha256": v_sha,
                "fps": 30.0,
                "frame_count": 300,
                "vehicles": [
                    {
                        "vehicle_id": "video:vid-streamline:vehicle-track:1",
                        "cabins": [
                            {
                                "cabin_id": "video:vid-streamline:vehicle-track:1:cabin:0",
                                "occupants": [
                                    {
                                        "occupant_id": "video:vid-streamline:vehicle-track:1:cabin:0:occupant-track:1",
                                        "role": "driver",
                                    }
                                ],
                            }
                        ],
                    }
                ],
            }
        ],
        output_path=roster_file,
    )
    manifest = extract_identity_manifest_from_roster(roster_file)
    manifest_file = tmp_path / "man.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lock_file = tmp_path / "man_lock.json"
    lock = freeze_identity_manifest(manifest_file, lock_file)
    manifest_sha = lock["manifest_sha256"]

    # Step 2: Generate skeleton with pre-bound manifest SHA
    skeleton = generate_annotation_skeleton(manifest, "vid-streamline", manifest_sha256=manifest_sha)
    assert skeleton["review_provenance"]["identity_manifest_sha256"] == manifest_sha

    # Step 3: Human annotator reviews and fills in event ground truth
    skeleton["review_provenance"]["status"] = "HUMAN_APPROVED"
    skeleton["review_provenance"]["reviewer_type"] = "HUMAN"
    skeleton["review_provenance"]["reviewer_id"] = "human-annotator-9"
    skeleton["review_provenance"]["reviewed_at"] = "2026-09-06T00:00:00Z"
    skeleton["review_provenance"]["evidence_hash"] = "e" * 64
    skeleton["review_provenance"]["adjudication_status"] = "FINAL"
    skeleton["occupants"][0]["reviewer_confirmed_role"] = True
    skeleton["occupants"][0]["role"] = "driver"
    skeleton["occupants"][0]["role_confidence"] = 1.0
    skeleton["occupants"][0]["vehicle_context_confirmed"] = True

    occ_id = skeleton["occupants"][0]["occupant_id"]
    skeleton["events"].append(
        {
            "event_id": "evt-phone-1",
            "event_type": "PHONE",
            "start_frame": 0,
            "end_frame": 100,
            "start_time_sec": 0.0,
            "end_time_sec": 100.0 / 30.0,
            "occupant_id": occ_id,
            "label": "PHONE_USE",
        }
    )
    # Split context interval into 2 valid contiguous intervals: [0, 150] with PHONE_USE, [150, 300] with NO_PHONE
    skeleton["context_intervals"] = [
        {
            "context_id": "ctx-1",
            "occupant_id": occ_id,
            "start_frame": 0,
            "end_frame": 150,
            "inside_vehicle": True,
            "outside_vehicle_person": False,
            "motorcycle_flag": False,
            "phone_state": "PHONE_USE",
            "seatbelt_state": "FASTENED",
            "visibility": "clear",
            "conditions": "daylight",
            "notes": "call active",
        },
        {
            "context_id": "ctx-2",
            "occupant_id": occ_id,
            "start_frame": 150,
            "end_frame": 300,
            "inside_vehicle": True,
            "outside_vehicle_person": False,
            "motorcycle_flag": False,
            "phone_state": "NO_PHONE",
            "seatbelt_state": "FASTENED",
            "visibility": "clear",
            "conditions": "daylight",
            "notes": "no call",
        },
    ]
    skeleton["context"]["inside_vehicle"] = True
    skeleton["context"]["outside_vehicle_person"] = False
    skeleton["context"]["motorcycle_flag"] = False

    ann_file = tmp_path / "ann.json"
    ann_file.write_text(json.dumps(skeleton, indent=2), encoding="utf-8")

    # Step 4: Validate via JSON Schema and semantic validator
    schema = load_schema("datasets/schemas/v2_event_sequence_annotation.schema.json")
    val_errors = validate_annotation(skeleton, schema)
    assert not val_errors, f"Schema/semantic validation failed: {val_errors}"

    # Step 5: Convert sequence to truth CSVs via process_file
    truth_csv = tmp_path / "truth.csv"
    context_csv = tmp_path / "context.csv"
    with truth_csv.open("w", encoding="utf-8", newline="") as ef, context_csv.open("w", encoding="utf-8", newline="") as cf:
        event_writer = csv.writer(ef)
        event_writer.writerow(EVENT_FIELDNAMES)
        context_writer = csv.writer(cf)
        context_writer.writerow(CONTEXT_FIELDNAMES)
        success = process_file(ann_file, event_writer, schema, context_writer=context_writer)
    assert success, "process_file failed to convert sequence to event and context CSVs"

    with truth_csv.open(newline="", encoding="utf-8-sig") as f:
        event_rows = list(csv.DictReader(f))
    assert len(event_rows) == 1
    assert event_rows[0]["identity_manifest_sha256"] == manifest_sha

    with context_csv.open(newline="", encoding="utf-8-sig") as f:
        context_rows = list(csv.DictReader(f))
    assert len(context_rows) == 2
    assert context_rows[0]["identity_manifest_sha256"] == manifest_sha
    assert context_rows[1]["identity_manifest_sha256"] == manifest_sha

    # Step 6: Freeze event ground truth & context ground truth
    ext_lock = tmp_path / "ext_lock.json"
    ext_lock.write_text(
        json.dumps(
            {
                "status": "FROZEN_EXTERNAL_TEST",
                "human_review_status": "ALL_APPROVED",
                "video_ids": ["vid-streamline"],
                "identity_manifest_sha256": manifest_sha,
                "videos": {
                    "vid-streamline": {
                        "sha256": v_sha,
                        "fps": 30.0,
                        "frame_count": 300,
                        "duration_seconds": 10.0,
                    }
                },
            }
        ),
        encoding="utf-8",
    )
    frozen_truth = freeze_event_ground_truth(
        truth_csv,
        ext_lock,
        tmp_path / "frozen_event_truth.json",
        identity_manifest_lock_path=lock_file,
    )
    assert frozen_truth["status"] == "FROZEN_EVENT_GROUND_TRUTH"
    assert frozen_truth["identity_manifest_sha256"] == manifest_sha

    frozen_context = freeze_context_ground_truth(
        context_csv,
        ext_lock,
        tmp_path / "frozen_context_truth.json",
        identity_manifest_lock_path=lock_file,
    )
    assert frozen_context["status"] == "FROZEN_CONTEXT_GROUND_TRUTH"
    assert frozen_context["identity_manifest_sha256"] == manifest_sha


def test_evaluate_frozen_scope_guardrails(tmp_path: Path):
    # Set up conditional manifest (RUNTIME_IDENTITY_TRACKS)
    manifest_lock_path = tmp_path / "conditional_manifest_lock.json"
    m_sha = "c" * 64
    manifest_lock = {
        "status": "FROZEN_IDENTITY_MANIFEST",
        "manifest_sha256": m_sha,
        "manifest_file": "man.json",
        "source_type": "RUNTIME_IDENTITY_TRACKS",
        "source_path": str(tmp_path / "tracks.jsonl"),
        "source_sha256": "s" * 64,
        "eligible_for_frozen_event_evaluation": True,
        "evaluation_scope": "CONDITIONAL_ON_SUCCESSFUL_OCCUPANT_TRACKING",
        "video_ids": ["vid-cond"],
        "videos": {"vid-cond": {"sha256": "v" * 64, "fps": 30.0, "frame_count": 300, "duration_seconds": 10.0}},
        "proven_identities": [
            {
                "video_id": "vid-cond",
                "vehicle_id": "video:vid-cond:vehicle-track:1",
                "cabin_id": "video:vid-cond:vehicle-track:1:cabin:0",
                "occupant_id": "video:vid-cond:vehicle-track:1:cabin:0:occupant-track:1",
            }
        ],
    }
    manifest_lock_path.write_text(json.dumps(manifest_lock), encoding="utf-8")

    truth_csv = tmp_path / "truth.csv"
    truth_row = {
        "video_id": "vid-cond",
        "event_id": "evt-1",
        "event_type": "PHONE",
        "start_seconds": "1.0",
        "end_seconds": "2.0",
        "occupant_id": "video:vid-cond:vehicle-track:1:cabin:0:occupant-track:1",
        "vehicle_id": "video:vid-cond:vehicle-track:1",
        "cabin_id": "video:vid-cond:vehicle-track:1:cabin:0",
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
        "reviewed_at": "2026-09-06T00:00:00Z",
        "adjudication_status": "FINAL",
        "notes": "test",
        "identity_manifest_sha256": m_sha,
    }
    with truth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(truth_row)

    ext_lock_path = tmp_path / "ext_lock.json"
    ext_lock = {
        "status": "FROZEN_EXTERNAL_TEST",
        "human_review_status": "ALL_APPROVED",
        "video_ids": ["vid-cond"],
        "identity_manifest_sha256": m_sha,
    }
    ext_lock_path.write_text(json.dumps(ext_lock), encoding="utf-8")

    event_lock_path = tmp_path / "event_lock.json"
    freeze_event_ground_truth(truth_csv, ext_lock_path, event_lock_path, identity_manifest_lock_path=manifest_lock_path)

    context_csv = tmp_path / "context.csv"
    context_row = {
        "video_id": "vid-cond",
        "context_id": "ctx-1",
        "occupant_id": "video:vid-cond:vehicle-track:1:cabin:0:occupant-track:1",
        "occupant_role": "driver",
        "vehicle_id": "video:vid-cond:vehicle-track:1",
        "cabin_id": "video:vid-cond:vehicle-track:1:cabin:0",
        "start_seconds": "0.0",
        "end_seconds": "10.0",
        "timeline_end_seconds": "10.0",
        "inside_vehicle": "true",
        "outside_vehicle_person": "false",
        "motorcycle_flag": "false",
        "phone_state": "PHONE_USE",
        "seatbelt_state": "FASTENED",
        "visibility": "clear",
        "conditions": "daylight",
        "human_review_status": "APPROVED",
        "reviewer_id": "rev-1",
        "reviewer_type": "HUMAN",
        "reviewed_at": "2026-09-06T00:00:00Z",
        "adjudication_status": "FINAL",
        "notes": "context test",
        "identity_manifest_sha256": m_sha,
    }
    with context_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(CONTEXT_REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(context_row)

    context_lock_path = tmp_path / "context_lock.json"
    freeze_context_ground_truth(context_csv, ext_lock_path, context_lock_path, identity_manifest_lock_path=manifest_lock_path)

    pred_csv = tmp_path / "preds.csv"
    pred_row = dict(truth_row)
    pred_row["observation_count"] = "5"
    pred_row["start_frame"] = "30"
    pred_row["end_frame"] = "60"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(pred_row.keys()))
        writer.writeheader()
        writer.writerow(pred_row)

    model_lock_path = tmp_path / "model_lock.json"
    model_lock_path.write_text(
        json.dumps(
            {
                "record_schema": "ROADWATCH_MODEL_VERSION_V2",
                "activation_state": "ACTIVE",
                "experiment_id": "exp-test",
                "locked_at": "2026-09-06T00:00:00Z",
                "weights_sha256": "w" * 64,
                "config_sha256": "c" * 64,
                "training_data_manifest_sha256": "t" * 64,
                "validation_metric_artifact": {"sha256": "v" * 64},
                "threshold_calibration_artifact": {"sha256": "k" * 64},
                "human_review_readiness_artifact": {"governed_training_ready": True},
                "code_commit": "abc1234",
            }
        ),
        encoding="utf-8",
    )

    out_json = tmp_path / "eval_out.json"

    # Default: hard reject conditional manifests
    with pytest.raises(ValueError, match="refusing frozen event evaluation: manifest evaluation_scope is 'CONDITIONAL_ON_SUCCESSFUL_OCCUPANT_TRACKING'"):
        evaluate_frozen(
            truth_csv,
            pred_csv,
            event_lock_path,
            model_lock_path,
            out_json,
            video_minutes=10.0 / 60.0,
            context_truth_path=context_csv,
            context_truth_lock_path=context_lock_path,
            allow_conditional_evaluation=False,
        )

    # Opt-in diagnostic conditional evaluation succeeds with explicit caution recorded
    out_cond_json = tmp_path / "eval_cond_out.json"
    report = evaluate_frozen(
        truth_csv,
        pred_csv,
        event_lock_path,
        model_lock_path,
        out_cond_json,
        video_minutes=10.0 / 60.0,
        context_truth_path=context_csv,
        context_truth_lock_path=context_lock_path,
        allow_conditional_evaluation=True,
    )
    assert report["status"] == "MEASURED_CONDITIONAL_DIAGNOSTIC"
    assert report["identity_manifest"]["evaluation_scope"] == "CONDITIONAL_ON_SUCCESSFUL_OCCUPANT_TRACKING"
    assert report["identity_manifest"]["conditional_evaluation_allowed"] is True
    assert "scope_caution" in report["identity_manifest"]
    integrity = verify_evaluation_integrity(out_cond_json)
    assert integrity["status"] == "FROZEN_EVENT_EVALUATION_INTEGRITY_VERIFIED"


def test_freeze_identity_adjudication_bijective_and_locality(tmp_path: Path):
    target_manifest_sha = "9" * 64
    valid_adjudication = {
        "human_review_status": "APPROVED",
        "reviewer_type": "HUMAN",
        "reviewer_id": "auditor-1",
        "reviewed_at": "2026-09-06T00:00:00Z",
        "target_identity_manifest_sha256": target_manifest_sha,
        "mappings": {
            "video:v1:vehicle-track:1:cabin:0:occupant-track:88": "video:v1:vehicle-track:1:cabin:0:occupant-track:1",
            "video:v1:vehicle-track:1:cabin:0:occupant-track:89": "video:v1:vehicle-track:1:cabin:0:occupant-track:2",
        },
    }
    adj_file = tmp_path / "adj.json"
    adj_file.write_text(json.dumps(valid_adjudication, indent=2), encoding="utf-8")
    lock_file = tmp_path / "adj_lock.json"
    lock = freeze_identity_adjudication(adj_file, lock_file)

    assert lock["status"] == "FROZEN_IDENTITY_ADJUDICATION"
    assert lock["human_review_status"] == "APPROVED"
    assert lock["mapping_count"] == 2
    assert len(lock["adjudication_sha256"]) == 64

    # Many-to-one mapping rejection
    many_to_one = dict(valid_adjudication)
    many_to_one["mappings"] = {
        "video:v1:vehicle-track:1:cabin:0:occupant-track:88": "video:v1:vehicle-track:1:cabin:0:occupant-track:1",
        "video:v1:vehicle-track:1:cabin:0:occupant-track:89": "video:v1:vehicle-track:1:cabin:0:occupant-track:1",  # Duplicate target!
    }
    m21_file = tmp_path / "m21.json"
    m21_file.write_text(json.dumps(many_to_one), encoding="utf-8")
    with pytest.raises(ValueError, match="many-to-one adjudication violation"):
        freeze_identity_adjudication(m21_file, tmp_path / "m21_lock.json")

    # Cross-cabin mapping rejection
    cross_cabin = dict(valid_adjudication)
    cross_cabin["mappings"] = {
        "video:v1:vehicle-track:1:cabin:0:occupant-track:88": "video:v1:vehicle-track:1:cabin:1:occupant-track:1",  # Cabin 0 -> Cabin 1!
    }
    xcab_file = tmp_path / "xcab.json"
    xcab_file.write_text(json.dumps(cross_cabin), encoding="utf-8")
    with pytest.raises(ValueError, match="cross-cabin adjudication violation"):
        freeze_identity_adjudication(xcab_file, tmp_path / "xcab_lock.json")

    # Non-human review rejection
    non_human = dict(valid_adjudication)
    non_human["reviewer_type"] = "AI"
    nh_file = tmp_path / "nh.json"
    nh_file.write_text(json.dumps(non_human), encoding="utf-8")
    with pytest.raises(ValueError, match="reviewer_type == 'HUMAN'"):
        freeze_identity_adjudication(nh_file, tmp_path / "nh_lock.json")


def test_evaluate_frozen_governed_identity_adjudication(tmp_path: Path):
    m_sha = "8" * 64
    manifest_lock_path = tmp_path / "full_manifest_lock.json"
    manifest_lock = {
        "status": "FROZEN_IDENTITY_MANIFEST",
        "manifest_sha256": m_sha,
        "manifest_file": "man.json",
        "source_type": "INDEPENDENT_IDENTITY_ROSTER",
        "source_path": str(tmp_path / "roster.json"),
        "source_sha256": "s" * 64,
        "eligible_for_frozen_event_evaluation": True,
        "evaluation_scope": "FULL_SYSTEM_EVENT_EVALUATION",
        "video_ids": ["vid-adj"],
        "videos": {"vid-adj": {"sha256": "v" * 64, "fps": 30.0, "frame_count": 300, "duration_seconds": 10.0}},
        "proven_identities": [
            {
                "video_id": "vid-adj",
                "vehicle_id": "video:vid-adj:vehicle-track:1",
                "cabin_id": "video:vid-adj:vehicle-track:1:cabin:0",
                "occupant_id": "video:vid-adj:vehicle-track:1:cabin:0:occupant-track:1",
            },
            {
                "video_id": "vid-adj",
                "vehicle_id": "video:vid-adj:vehicle-track:1",
                "cabin_id": "video:vid-adj:vehicle-track:1:cabin:0",
                "occupant_id": "video:vid-adj:vehicle-track:1:cabin:0:occupant-track:77",
            },
        ],
    }
    manifest_lock_path.write_text(json.dumps(manifest_lock), encoding="utf-8")

    truth_csv = tmp_path / "truth.csv"
    truth_row = {
        "video_id": "vid-adj",
        "event_id": "evt-1",
        "event_type": "PHONE",
        "start_seconds": "1.0",
        "end_seconds": "2.0",
        "occupant_id": "video:vid-adj:vehicle-track:1:cabin:0:occupant-track:1",
        "vehicle_id": "video:vid-adj:vehicle-track:1",
        "cabin_id": "video:vid-adj:vehicle-track:1:cabin:0",
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
        "reviewed_at": "2026-09-06T00:00:00Z",
        "adjudication_status": "FINAL",
        "notes": "test",
        "identity_manifest_sha256": m_sha,
    }
    with truth_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(truth_row)

    ext_lock_path = tmp_path / "ext_lock.json"
    ext_lock_path.write_text(
        json.dumps(
            {
                "status": "FROZEN_EXTERNAL_TEST",
                "human_review_status": "ALL_APPROVED",
                "video_ids": ["vid-adj"],
                "identity_manifest_sha256": m_sha,
            }
        ),
        encoding="utf-8",
    )

    event_lock_path = tmp_path / "event_lock.json"
    freeze_event_ground_truth(truth_csv, ext_lock_path, event_lock_path, identity_manifest_lock_path=manifest_lock_path)

    context_csv = tmp_path / "context.csv"
    context_row_1 = {
        "video_id": "vid-adj",
        "context_id": "ctx-1",
        "occupant_id": "video:vid-adj:vehicle-track:1:cabin:0:occupant-track:1",
        "occupant_role": "driver",
        "vehicle_id": "video:vid-adj:vehicle-track:1",
        "cabin_id": "video:vid-adj:vehicle-track:1:cabin:0",
        "start_seconds": "0.0",
        "end_seconds": "10.0",
        "timeline_end_seconds": "10.0",
        "inside_vehicle": "true",
        "outside_vehicle_person": "false",
        "motorcycle_flag": "false",
        "phone_state": "PHONE_USE",
        "seatbelt_state": "FASTENED",
        "visibility": "clear",
        "conditions": "daylight",
        "human_review_status": "APPROVED",
        "reviewer_id": "rev-1",
        "reviewer_type": "HUMAN",
        "reviewed_at": "2026-09-06T00:00:00Z",
        "adjudication_status": "FINAL",
        "notes": "context test",
        "identity_manifest_sha256": m_sha,
    }
    context_row_77 = dict(context_row_1)
    context_row_77["context_id"] = "ctx-77"
    context_row_77["occupant_id"] = "video:vid-adj:vehicle-track:1:cabin:0:occupant-track:77"
    with context_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(CONTEXT_REQUIRED_COLUMNS))
        writer.writeheader()
        writer.writerow(context_row_1)
        writer.writerow(context_row_77)

    context_lock_path = tmp_path / "context_lock.json"
    freeze_context_ground_truth(context_csv, ext_lock_path, context_lock_path, identity_manifest_lock_path=manifest_lock_path)

    # Model prediction has runtime track ID 77
    pred_csv = tmp_path / "pred_77.csv"
    pred_row = dict(truth_row)
    pred_row["occupant_id"] = "video:vid-adj:vehicle-track:1:cabin:0:occupant-track:77"
    pred_row["observation_count"] = "5"
    pred_row["start_frame"] = "30"
    pred_row["end_frame"] = "60"
    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=sorted(pred_row.keys()))
        writer.writeheader()
        writer.writerow(pred_row)

    model_lock_path = tmp_path / "model_lock.json"
    model_lock_path.write_text(
        json.dumps(
            {
                "record_schema": "ROADWATCH_MODEL_VERSION_V2",
                "activation_state": "ACTIVE",
                "experiment_id": "exp-adj",
                "locked_at": "2026-09-06T00:00:00Z",
                "weights_sha256": "w" * 64,
                "config_sha256": "c" * 64,
                "training_data_manifest_sha256": "t" * 64,
                "validation_metric_artifact": {"sha256": "v" * 64},
                "threshold_calibration_artifact": {"sha256": "k" * 64},
                "human_review_readiness_artifact": {"governed_training_ready": True},
                "code_commit": "abc1234",
            }
        ),
        encoding="utf-8",
    )

    # Un-adjudicated evaluate_frozen produces 0 TP, 1 Missed Event
    unadj_report_path = tmp_path / "unadj_report.json"
    unadj_report = evaluate_frozen(
        truth_csv,
        pred_csv,
        event_lock_path,
        model_lock_path,
        unadj_report_path,
        video_minutes=10.0 / 60.0,
        context_truth_path=context_csv,
        context_truth_lock_path=context_lock_path,
    )
    assert unadj_report["event_types"]["PHONE"]["true_positives"] == 0
    assert unadj_report["event_types"]["PHONE"]["missed_events"] == 1

    # Attempting to pass raw unfrozen mapping raises ValueError
    raw_adj_path = tmp_path / "raw_adj.json"
    raw_adj_path.write_text(json.dumps({"mappings": {"video:vid-adj:vehicle-track:1:cabin:0:occupant-track:77": "video:vid-adj:vehicle-track:1:cabin:0:occupant-track:1"}}))
    with pytest.raises(ValueError, match="must be a FROZEN_IDENTITY_ADJUDICATION artifact"):
        evaluate_frozen(
            truth_csv,
            pred_csv,
            event_lock_path,
            model_lock_path,
            tmp_path / "err.json",
            video_minutes=10.0 / 60.0,
            context_truth_path=context_csv,
            context_truth_lock_path=context_lock_path,
            identity_adjudication_path=raw_adj_path,
        )

    # Create approved adjudication and freeze it
    valid_adj = {
        "human_review_status": "APPROVED",
        "reviewer_type": "HUMAN",
        "reviewer_id": "auditor-42",
        "reviewed_at": "2026-09-06T00:00:00Z",
        "target_identity_manifest_sha256": m_sha,
        "mappings": {
            "video:vid-adj:vehicle-track:1:cabin:0:occupant-track:77": "video:vid-adj:vehicle-track:1:cabin:0:occupant-track:1"
        },
    }
    adj_in = tmp_path / "adj_in.json"
    adj_in.write_text(json.dumps(valid_adj, indent=2))
    adj_lock = tmp_path / "adj_lock.json"
    freeze_identity_adjudication(adj_in, adj_lock, identity_manifest_lock_path=manifest_lock_path)

    # Evaluated with frozen adjudication -> 1 TP, 0 Missed Events
    adj_report_path = tmp_path / "adj_report.json"
    adj_report = evaluate_frozen(
        truth_csv,
        pred_csv,
        event_lock_path,
        model_lock_path,
        adj_report_path,
        video_minutes=10.0 / 60.0,
        context_truth_path=context_csv,
        context_truth_lock_path=context_lock_path,
        identity_adjudication_path=adj_lock,
    )
    assert adj_report["event_types"]["PHONE"]["true_positives"] == 1
    assert adj_report["event_types"]["PHONE"]["missed_events"] == 0
    assert "identity_adjudication_lock" in adj_report
    assert adj_report["identity_adjudication_lock"]["reviewer_id"] == "auditor-42"

    # Integrity verification verifies the adjudication lock artifact as well
    integrity = verify_evaluation_integrity(adj_report_path)
    assert integrity["status"] == "FROZEN_EVENT_EVALUATION_INTEGRITY_VERIFIED"
    assert "identity-adjudication lock" in integrity["verified_artifacts"]


def test_freeze_external_test_bijective_video_coverage(tmp_path: Path):
    policy_path = tmp_path / "policy.yaml"
    policy_path.write_text(
        """schema_version: 1
dataset_role: EXTERNAL_TEST
require_identity_manifest: true
required_fields:
  - sample_id
  - dataset_role
  - source_id
  - camera_id
  - video_id
  - vehicle_id
  - person_id
  - video_path
  - sha256
  - annotation_path
  - annotation_sha256
  - conditions
  - human_review_status
  - reviewer_id
  - reviewer_type
  - reviewed_at
required_condition_coverage: [daylight]
disjoint_dimensions: [sha256, source_id, camera_id, video_id, vehicle_id, person_id]
minimum_independent_groups: {source_id: 1, camera_id: 1, video_id: 1, vehicle_id: 1, person_id: 1}
""",
        encoding="utf-8",
    )

    v1 = tmp_path / "v1.mp4"
    v1.write_bytes(b"video 1")
    v1_sha = sha256_file(v1)

    ann1 = tmp_path / "ann1.json"
    ann1.write_text("[]")
    ann1_sha = sha256_file(ann1)

    record1 = {
        "sample_id": "s1",
        "dataset_role": "EXTERNAL_TEST",
        "source_id": "src1",
        "camera_id": "cam1",
        "video_id": "vid-1",
        "vehicle_id": "veh1",
        "person_id": "p1",
        "video_path": str(v1.resolve()),
        "sha256": v1_sha,
        "annotation_path": str(ann1.resolve()),
        "annotation_sha256": ann1_sha,
        "conditions": ["daylight"],
        "human_review_status": "APPROVED",
        "reviewer_id": "rev1",
        "reviewer_type": "HUMAN",
        "reviewed_at": "2026-09-05T00:00:00Z",
    }
    ext_manifest_path = tmp_path / "ext_manifest.jsonl"
    ext_manifest_path.write_text(json.dumps(record1) + "\n")
    dev_manifest_path = tmp_path / "dev_manifest.jsonl"
    dev_manifest_path.write_text("")

    # Identity manifest lock has vid-1 AND extra vid-2
    extra_vid_lock = {
        "status": "FROZEN_IDENTITY_MANIFEST",
        "manifest_sha256": "m" * 64,
        "manifest_file": "man.json",
        "source_type": "INDEPENDENT_IDENTITY_ROSTER",
        "source_path": str(v1.resolve()),
        "source_sha256": v1_sha,
        "eligible_for_frozen_event_evaluation": True,
        "evaluation_scope": "FULL_SYSTEM_EVENT_EVALUATION",
        "video_ids": ["vid-1", "vid-2"],  # Extra video!
        "videos": {
            "vid-1": {"sha256": v1_sha, "fps": 30.0, "frame_count": 300, "duration_seconds": 10.0},
            "vid-2": {"sha256": "2" * 64, "fps": 30.0, "frame_count": 300, "duration_seconds": 10.0},
        },
        "proven_identities": [],
    }
    extra_vid_lock_path = tmp_path / "extra_vid_lock.json"
    extra_vid_lock_path.write_text(json.dumps(extra_vid_lock))

    with pytest.raises(ValueError, match="frozen identity manifest covers videos not present in external test manifest: \\['vid-2'\\]"):
        freeze_external_test(
            ext_manifest_path,
            dev_manifest_path,
            policy_path,
            tmp_path / "frozen_extra.json",
            identity_manifest_lock_path=extra_vid_lock_path,
        )


