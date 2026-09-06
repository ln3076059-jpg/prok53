import json
from pathlib import Path

import pytest

from training.common import sha256_file
from training.freeze_external_test import freeze_external_test
from training.validate_sequence_intake import (
    DISJOINT_DIMENSIONS,
    read_intake,
    validate_sequence_intake,
)


@pytest.fixture
def intake(tmp_path):
    video = tmp_path / "video.mp4"
    video.write_bytes(b"synthetic-test-fixture-not-real-video")
    row = {
        "proposed_role": "NEW_UNTOUCHED_HOLDOUT",
        "video_path": str(video), "sha256": sha256_file(video),
        "source_id": "source-h", "camera_id": "camera-h", "video_id": "video-h",
        "capture_session_id": "session-h", "physical_vehicle_group_id": "vehicle-h",
        "person_group_ids": ["person-h1", "person-h2"],
        "prior_usage": "NEVER_USED", "model_predictions_seen": False,
    }
    for kind in ("rights", "independence"):
        path = tmp_path / f"{kind}.txt"
        path.write_text("Synthetic test evidence only", encoding="utf-8")
        row[f"{kind}_evidence_path"] = str(path)
        row[f"{kind}_evidence_sha256"] = sha256_file(path)
    dev = {dimension: f"development-{dimension}" for dimension in DISJOINT_DIMENSIONS}
    dev["sha256"] = "f" * 64
    dev["person_group_ids"] = ["person-d"]
    return row, dev


@pytest.mark.parametrize("dimension", DISJOINT_DIMENSIONS)
def test_intake_rejects_each_physical_and_source_overlap(intake, dimension):
    row, dev = intake
    dev[dimension] = row[dimension]
    if dimension == "person_group_ids":
        dev[dimension] = ["unrelated", "person-h2"]
    report = validate_sequence_intake([row], [dev])
    assert report["status"] == "NOT_READY_FOR_UNTOUCHED_FREEZE"
    assert report["development_overlap"][dimension]


@pytest.mark.parametrize("value", [None, "UNKNOWN", "true", True, 0, 1, "", "False"])
def test_intake_requires_explicit_false(intake, value):
    row, dev = intake
    row["model_predictions_seen"] = value
    assert any("model_predictions_seen" in e for e in validate_sequence_intake([row], [dev])["errors"])


@pytest.mark.parametrize("value", [None, "UNKNOWN", "TRAIN", "CALIBRATION", "TEST", ""])
def test_intake_requires_unused_history(intake, value):
    row, dev = intake
    row["prior_usage"] = value
    assert any("prior_usage" in e for e in validate_sequence_intake([row], [dev])["errors"])


@pytest.mark.parametrize("role", ["holdout", "development"])
@pytest.mark.parametrize("dimension", ["capture_session_id", "physical_vehicle_group_id", "person_group_ids"])
@pytest.mark.parametrize("value", [None, "UNKNOWN", "NOT_PROVABLE", "video:A:occupant-track:1"])
def test_intake_requires_physical_identity_on_both_sides(intake, role, dimension, value):
    row, dev = intake
    (row if role == "holdout" else dev)[dimension] = value
    assert validate_sequence_intake([row], [dev])["status"] == "NOT_READY_FOR_UNTOUCHED_FREEZE"


@pytest.mark.parametrize("kind", ["rights", "independence"])
@pytest.mark.parametrize("mutation", ["missing", "empty", "changed", "missing_hash"])
def test_intake_requires_bound_evidence(intake, kind, mutation):
    row, dev = intake
    path = Path(row[f"{kind}_evidence_path"])
    if mutation == "missing":
        path.unlink()
    elif mutation == "missing_hash":
        row.pop(f"{kind}_evidence_sha256")
    else:
        path.write_text("" if mutation == "empty" else "changed", encoding="utf-8")
    assert validate_sequence_intake([row], [dev])["status"] == "NOT_READY_FOR_UNTOUCHED_FREEZE"


def test_csv_person_sets_and_empty_comparator_fail_closed(intake, tmp_path):
    import csv

    row, dev = intake
    row["model_predictions_seen"] = "false"
    row["person_group_ids"] = "person-h1;person-h2"
    path = tmp_path / "intake.csv"
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    parsed = read_intake(path)
    assert validate_sequence_intake(parsed, [dev])["status"] == "SEQUENCE_INTAKE_CHECKS_PASSED"
    assert validate_sequence_intake(parsed, [])["status"] == "NOT_READY_FOR_UNTOUCHED_FREEZE"
    dev["person_group_ids"] = "person-h2;person-d"
    assert validate_sequence_intake(parsed, [dev])["development_overlap"]["person_group_ids"] == ["person-h2"]


def test_cli_records_not_ready_and_does_not_overwrite(tmp_path, monkeypatch):
    from training.validate_sequence_intake import main

    holdout = tmp_path / "holdout.csv"
    development = tmp_path / "development.jsonl"
    output = tmp_path / "check.json"
    holdout.write_text("video_id,person_group_ids\n", encoding="utf-8")
    development.write_text("", encoding="utf-8")
    monkeypatch.setattr("sys.argv", [
        "validate_sequence_intake", str(holdout), str(development), "--output", str(output),
    ])
    with pytest.raises(SystemExit) as exc:
        main()
    assert exc.value.code == 1
    report = json.loads(output.read_text(encoding="utf-8"))
    assert report["status"] == "NOT_READY_FOR_UNTOUCHED_FREEZE"
    with pytest.raises(FileExistsError):
        main()


@pytest.fixture
def official_freeze(intake, tmp_path):
    import yaml

    row, dev = intake
    policy = Path("datasets/v2_external_test_policy.yaml")
    settings = yaml.safe_load(policy.read_text(encoding="utf-8"))
    annotation = tmp_path / "events.json"
    annotation.write_text("[]", encoding="utf-8")
    records = []
    for index in range(12):
        video = tmp_path / f"holdout-{index}.mp4"
        video.write_bytes(f"synthetic test fixture {index}".encode())
        records.append({
            **row, "sample_id": f"sample-{index}", "dataset_role": "EXTERNAL_TEST",
            "video_id": f"h-{index}", "video_path": str(video), "sha256": sha256_file(video),
            "source_id": f"source-{index % 2}", "camera_id": f"camera-{index % 3}",
            "vehicle_id": f"video:h-{index}:provided-vehicle", "person_id": f"local-{index}",
            "physical_vehicle_group_id": f"physical-vehicle-{index % 8}",
            "person_group_ids": [f"physical-person-{index % 8}"],
            "annotation_path": str(annotation), "annotation_sha256": sha256_file(annotation),
            "conditions": settings["required_condition_coverage"],
            "human_review_status": "APPROVED", "reviewer_type": "HUMAN",
            "reviewer_id": "test-reviewer", "reviewed_at": "2026-09-06T00:00:00Z",
        })
    manifest = tmp_path / "external.jsonl"
    development = tmp_path / "development.jsonl"
    development.write_text(json.dumps(dev) + "\n", encoding="utf-8")
    lock = {
        "status": "FROZEN_IDENTITY_MANIFEST", "manifest_sha256": "a" * 64,
        "eligible_for_frozen_event_evaluation": True,
        "evaluation_scope": "FULL_SYSTEM_EVENT_EVALUATION",
        "video_ids": [r["video_id"] for r in records],
        "videos": {r["video_id"]: {"sha256": r["sha256"]} for r in records},
    }
    lock_path = tmp_path / "identity.json"
    return records, manifest, development, policy, tmp_path / "frozen.json", lock, lock_path


@pytest.mark.parametrize("case", [
    "valid", "session_overlap", "vehicle_overlap", "person_overlap", "seen", "used",
    "missing_development_identity", "conditional", "physical_count", "evidence_tamper",
])
def test_official_freezer_enforces_intake_without_precheck(official_freeze, case):
    records, manifest, development, policy, output, lock, lock_path = official_freeze
    dev = json.loads(development.read_text(encoding="utf-8"))
    if case.endswith("_overlap"):
        dimension = {"session_overlap": "capture_session_id", "vehicle_overlap": "physical_vehicle_group_id",
                     "person_overlap": "person_group_ids"}[case]
        records[0][dimension] = dev[dimension]
    elif case == "seen":
        records[0]["model_predictions_seen"] = "UNKNOWN"
    elif case == "used":
        records[0]["prior_usage"] = "CALIBRATION"
    elif case == "missing_development_identity":
        dev.pop("person_group_ids")
    elif case == "conditional":
        lock["evaluation_scope"] = "CONDITIONAL_ON_SUCCESSFUL_OCCUPANT_TRACKING"
    elif case == "physical_count":
        for record in records:
            record["physical_vehicle_group_id"] = "one-real-car"
            record["person_group_ids"] = ["one-real-person"]
    elif case == "evidence_tamper":
        Path(records[0]["independence_evidence_path"]).write_text("modified", encoding="utf-8")
    manifest.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
    development.write_text(json.dumps(dev) + "\n", encoding="utf-8")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    if case != "valid":
        with pytest.raises(ValueError, match="cannot freeze external test"):
            freeze_external_test(manifest, development, policy, output, lock_path)
        assert not output.exists()
    else:
        result = freeze_external_test(manifest, development, policy, output, lock_path)
        assert result["sequence_intake_validation"]["status"] == "SEQUENCE_INTAKE_CHECKS_PASSED"
        assert result["independent_group_coverage"]["person_group_ids"]["actual"] == 8
        assert result["independent_group_coverage"]["physical_vehicle_group_id"]["actual"] == 8
        assert len(result["sequence_intake_validation"]["evidence_files"]) == 24
