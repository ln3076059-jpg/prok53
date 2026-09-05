from __future__ import annotations

import json
from pathlib import Path
import jsonschema
import pytest

from training.extract_identity_manifest import generate_annotation_skeleton
from training.identity_contract import validate_identity_contract


def test_generate_annotation_skeleton_tracks():
    skeleton = generate_annotation_skeleton(
        video_id="video-99",
        vehicle_track_id=2,
        cabin_epoch=0,
        occupant_track_ids=[5, 6],
        fps=30.0,
        frame_count=600,
        source_id="cam-front",
    )

    assert skeleton["video_id"] == "video-99"
    assert skeleton["vehicle_id"] == "video:video-99:vehicle-track:2"
    assert skeleton["cabin_id"] == "video:video-99:vehicle-track:2:cabin:0"
    assert len(skeleton["occupants"]) == 2
    assert skeleton["occupants"][0]["occupant_id"] == "video:video-99:vehicle-track:2:cabin:0:occupant-track:5"
    assert skeleton["occupants"][1]["occupant_id"] == "video:video-99:vehicle-track:2:cabin:0:occupant-track:6"
    assert skeleton["events"] == []
    assert skeleton["review_provenance"]["reviewer_type"] == "AI"
    assert skeleton["review_provenance"]["status"] == "AI_REVIEWED_PROPOSAL"

    # Strict contract validation
    for occ in skeleton["occupants"]:
        errs = validate_identity_contract(
            skeleton["video_id"],
            skeleton["vehicle_id"],
            skeleton["cabin_id"],
            occ["occupant_id"],
        )
        assert errs == []


def test_generate_annotation_skeleton_provided_cabin():
    skeleton = generate_annotation_skeleton(
        video_id="video-single",
        occupant_track_ids=[1],
        is_provided_cabin=True,
    )

    assert skeleton["vehicle_id"] == "video:video-single:provided-vehicle"
    assert skeleton["cabin_id"] == "video:video-single:provided-cabin"
    assert skeleton["occupants"][0]["occupant_id"] == "video:video-single:provided-cabin:occupant-track:1"

    errs = validate_identity_contract(
        skeleton["video_id"],
        skeleton["vehicle_id"],
        skeleton["cabin_id"],
        skeleton["occupants"][0]["occupant_id"],
    )
    assert errs == []


def test_skeleton_conforms_to_sequence_schema():
    schema_path = Path("datasets/schemas/v2_event_sequence_annotation.schema.json")
    assert schema_path.exists()
    schema = json.loads(schema_path.read_text(encoding="utf-8"))

    skeleton = generate_annotation_skeleton(
        video_id="video-schema-test",
        vehicle_track_id=1,
        occupant_track_ids=[10],
    )

    # Validates against the official JSON schema
    jsonschema.validate(instance=skeleton, schema=schema)


def test_cli_execution(tmp_path, monkeypatch):
    from training.extract_identity_manifest import main

    out_file = tmp_path / "skeleton.json"
    test_args = [
        "extract_identity_manifest.py",
        "--video-id", "video-cli",
        "--vehicle-track-id", "3",
        "--cabin-epoch", "1",
        "--occupant-tracks", "7", "8",
        "--output", str(out_file),
    ]
    monkeypatch.setattr("sys.argv", test_args)
    main()

    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8"))
    assert data["video_id"] == "video-cli"
    assert data["vehicle_id"] == "video:video-cli:vehicle-track:3"
    assert data["cabin_id"] == "video:video-cli:vehicle-track:3:cabin:1"
    assert len(data["occupants"]) == 2
