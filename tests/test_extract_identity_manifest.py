from __future__ import annotations

import csv
import json
from pathlib import Path
import jsonschema
import pytest

from training.extract_identity_manifest import (
    extract_identities_from_predictions_csv,
    extract_identity_manifest_from_annotations,
    extract_identity_manifest_from_tracks,
    freeze_identity_manifest,
    generate_annotation_skeleton,
    generate_annotation_skeletons_for_video,
)
from training.evaluate_events import evaluate, evaluate_frozen
from training.freeze_external_test import freeze_external_test
from training.build_event_truth_from_sequences import process_file
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
    assert manifest["evaluation_scope"] == "FULL_SYSTEM_EVENT_EVALUATION"
    assert manifest["eligible_for_frozen_event_evaluation"] is True
    assert len(manifest["proven_identities"]) == 2
    assert manifest["videos"]["vid-annotated"]["duration_seconds"] == 100.0

    manifest_file = tmp_path / "manifest.json"
    manifest_file.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    lock_file = tmp_path / "manifest_lock.json"
    lock = freeze_identity_manifest(manifest_file, lock_file)

    assert lock["status"] == "FROZEN_IDENTITY_MANIFEST"
    assert lock["eligible_for_frozen_event_evaluation"] is True
    assert lock["evaluation_scope"] == "FULL_SYSTEM_EVENT_EVALUATION"
    assert lock["videos"]["vid-annotated"]["sha256"] == "v" * 64


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

    # Identity manifest lock with MISMATCHED video SHA
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
                "sha256": "different_sha_from_tracking" + "0" * 36,
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


