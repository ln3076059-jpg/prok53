import json
from pathlib import Path

import pytest

from training.common import sha256_file
from training.freeze_external_test import freeze_external_test
from training.freeze_development_lineage import COVERED_USAGE, freeze_development_lineage
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


def freeze_test_lineage(path):
    evidence = path.with_suffix(".evidence.txt")
    evidence.write_text("Synthetic completeness evidence for tests only", encoding="utf-8")
    review = path.with_suffix(".review.json")
    review.write_text(json.dumps({
        "lineage_sha256": sha256_file(path), "completeness_status": "COMPLETE",
        "covered_usage": sorted(COVERED_USAGE),
        "human_review_status": "APPROVED", "reviewer_type": "HUMAN",
        "reviewer_id": "test-auditor", "reviewed_at": "2026-09-06T00:00:00Z",
        "adjudication_status": "FINAL",
        "completeness_evidence": {"path": str(evidence), "sha256": sha256_file(evidence)},
    }), encoding="utf-8")
    lock = path.with_suffix(".lock.json")
    freeze_development_lineage(path, review, lock)
    return lock


def canonical_sequence(item):
    cabin = f"video:{item['video_id']}:provided-cabin"
    occupant = f"{cabin}:occupant-track:1"
    return {
        "sequence_id": f"seq-{item['video_id']}", "video_id": item["video_id"],
        "video_sha256": item["sha256"], "vehicle_id": item["vehicle_id"], "cabin_id": cabin,
        "source_id": item["source_id"], "camera_id": item["camera_id"],
        "fps": 30.0, "frame_count": 300,
        "start_time": "2026-09-06T00:00:00Z", "end_time": "2026-09-06T00:00:10Z",
        "occupants": [{"occupant_id": occupant, "role": "driver"}],
        "events": [{"event_type": "PHONE", "occupant_id": occupant,
                    "start_frame": 0, "end_frame": 299, "label": "PHONE_USE"}],
        "context_intervals": [{
            "context_id": f"ctx-{item['video_id']}", "occupant_id": occupant,
            "start_frame": 0, "end_frame": 300, "inside_vehicle": True,
            "outside_vehicle_person": False, "motorcycle_flag": False,
            "phone_state": "PHONE_USE", "seatbelt_state": "FASTENED",
            "visibility": "clear", "conditions": "daylight",
        }],
        "context": {"inside_vehicle": True, "outside_vehicle_person": False, "motorcycle_flag": False},
        "review_provenance": {
            "reviewer_type": "HUMAN", "status": "HUMAN_APPROVED", "reviewer_id": "test-auditor",
            "reviewed_at": "2026-09-06T00:00:00Z", "adjudication_status": "FINAL",
            "annotation_version": "v2.0", "identity_manifest_sha256": "a" * 64,
            "evidence_hash": "b" * 64,
        },
    }


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
    freeze_test_lineage(development)
    for record in records:
        annotation = tmp_path / f"sequence-{record['video_id']}.json"
        annotation.write_text(json.dumps(canonical_sequence(record)), encoding="utf-8")
        record["annotation_path"] = str(annotation)
        record["annotation_sha256"] = sha256_file(annotation)
    lock = {
        "status": "FROZEN_IDENTITY_MANIFEST", "manifest_sha256": "a" * 64,
        "eligible_for_frozen_event_evaluation": True,
        "evaluation_scope": "FULL_SYSTEM_EVENT_EVALUATION",
        "video_ids": [r["video_id"] for r in records],
        "videos": {r["video_id"]: {"sha256": r["sha256"], "fps": 30.0, "frame_count": 300} for r in records},
        "proven_identities": [{
            "video_id": r["video_id"], "vehicle_id": r["vehicle_id"],
            "cabin_id": f"video:{r['video_id']}:provided-cabin",
            "occupant_id": f"video:{r['video_id']}:provided-cabin:occupant-track:1",
        } for r in records],
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
            freeze_external_test(manifest, development, policy, output, lock_path,
                                 development.with_suffix(".lock.json"))
        assert not output.exists()
    else:
        result = freeze_external_test(manifest, development, policy, output, lock_path,
                                      development.with_suffix(".lock.json"))
        assert result["sequence_intake_validation"]["status"] == "SEQUENCE_INTAKE_CHECKS_PASSED"
        assert result["independent_group_coverage"]["person_group_ids"]["actual"] == 8
        assert result["independent_group_coverage"]["physical_vehicle_group_id"]["actual"] == 8
        assert len(result["sequence_intake_validation"]["evidence_files"]) == 24


@pytest.mark.parametrize("mutation", [
    "legacy_list", "gap", "ai", "pending", "wrong_video", "wrong_hash", "wrong_fps",
    "wrong_manifest", "missing_occupant", "file_changed", "extra_file_changed", "blank_reviewer",
])
def test_official_freeze_rejects_invalid_canonical_truth(official_freeze, mutation):
    rows, manifest, development, policy, output, lock, lock_path = official_freeze
    row = rows[0]
    path = Path(row["annotation_path"])
    annotation = json.loads(path.read_text(encoding="utf-8"))
    if mutation == "legacy_list":
        annotation = []
    elif mutation == "gap":
        annotation["context_intervals"][0]["start_frame"] = 1
    elif mutation == "ai":
        annotation["review_provenance"]["reviewer_type"] = "AI"
        annotation["review_provenance"]["status"] = "AI_REVIEWED_PROPOSAL"
    elif mutation == "pending":
        annotation["review_provenance"]["adjudication_status"] = "PENDING"
    elif mutation == "wrong_video":
        annotation["video_id"] = "different-video"
    elif mutation == "wrong_hash":
        annotation["video_sha256"] = "f" * 64
    elif mutation == "wrong_fps":
        annotation["fps"] = 25.0
    elif mutation == "wrong_manifest":
        annotation["review_provenance"]["identity_manifest_sha256"] = "f" * 64
    elif mutation == "missing_occupant":
        annotation["occupants"] = []
        annotation["events"] = []
        annotation["context_intervals"] = []
    elif mutation == "file_changed":
        annotation["context_intervals"][0]["notes"] = "tampered"
    elif mutation == "extra_file_changed":
        extra_path = path.with_name("extra-tampered.json")
        extra_path.write_bytes(path.read_bytes())
        row["additional_sequence_annotations"] = [{"path": str(extra_path), "sha256": "f" * 64}]
    elif mutation == "blank_reviewer":
        annotation["review_provenance"]["reviewer_id"] = " "
    path.write_text(json.dumps(annotation), encoding="utf-8")
    if mutation != "file_changed":
        row["annotation_sha256"] = sha256_file(path)
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="cannot freeze external test"):
        freeze_external_test(manifest, development, policy, output, lock_path,
                             development.with_suffix(".lock.json"))
    assert not output.exists()


@pytest.mark.parametrize("multiple_cabins", [False, True])
def test_canonical_truth_flows_through_external_event_and_context_freeze(
    official_freeze, tmp_path, multiple_cabins,
):
    import csv

    from training.build_event_truth_from_sequences import (
        CONTEXT_FIELDNAMES, EVENT_FIELDNAMES, process_file,
    )
    from training.freeze_context_ground_truth import freeze_context_ground_truth
    from training.freeze_event_ground_truth import freeze_event_ground_truth
    from training.validate_event_sequence_annotations import load_schema

    rows, manifest, development, policy, output, lock, lock_path = official_freeze
    if multiple_cabins:
        row = rows[0]
        extra = canonical_sequence(row)
        vehicle = f"video:{row['video_id']}:vehicle-track:2"
        cabin = f"{vehicle}:cabin:0"
        occupant = f"{cabin}:occupant-track:1"
        extra.update(sequence_id="extra-cabin", vehicle_id=vehicle, cabin_id=cabin)
        extra["occupants"][0]["occupant_id"] = occupant
        extra["events"][0]["occupant_id"] = occupant
        extra["context_intervals"][0].update(occupant_id=occupant, context_id="extra-context")
        path = tmp_path / "extra-cabin.json"
        path.write_text(json.dumps(extra), encoding="utf-8")
        row["additional_sequence_annotations"] = [{"path": str(path), "sha256": sha256_file(path)}]
        lock["proven_identities"].append({
            "video_id": row["video_id"], "vehicle_id": vehicle,
            "cabin_id": cabin, "occupant_id": occupant,
        })
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    external = freeze_external_test(manifest, development, policy, output, lock_path,
                                    development.with_suffix(".lock.json"))
    event_csv = tmp_path / "event.csv"
    context_csv = tmp_path / "context.csv"
    schema = load_schema("datasets/schemas/v2_event_sequence_annotation.schema.json")
    with event_csv.open("w", newline="", encoding="utf-8") as ef, context_csv.open("w", newline="", encoding="utf-8") as cf:
        ew = csv.DictWriter(ef, fieldnames=EVENT_FIELDNAMES)
        cw = csv.DictWriter(cf, fieldnames=CONTEXT_FIELDNAMES)
        ew.writeheader()
        cw.writeheader()
        for references in external["sequence_annotations"].values():
            for reference in references:
                assert process_file(Path(reference["path"]), ew, schema, cw)
    events = freeze_event_ground_truth(event_csv, output, tmp_path / "event-lock.json")
    context = freeze_context_ground_truth(context_csv, output, tmp_path / "context-lock.json")
    assert events["status"] == "FROZEN_EVENT_GROUND_TRUTH"
    assert context["coverage_status"] == "FULL_TIMELINE_NO_GAPS_OR_OVERLAPS"
    assert sum(len(refs) for refs in external["sequence_annotations"].values()) == 12 + multiple_cabins


@pytest.mark.parametrize("mutation", [
    "no_lock", "changed_lineage", "changed_review", "changed_evidence", "wrong_lock",
])
def test_official_freeze_requires_exact_development_lineage(official_freeze, mutation, tmp_path):
    rows, manifest, development, policy, output, lock, lock_path = official_freeze
    lineage_lock = development.with_suffix(".lock.json")
    if mutation == "no_lock":
        lineage_lock = None
    elif mutation == "changed_lineage":
        development.write_bytes(development.read_bytes() + b"\n")
    elif mutation == "changed_review":
        review = development.with_suffix(".review.json")
        review.write_bytes(review.read_bytes() + b"\n")
    elif mutation == "changed_evidence":
        development.with_suffix(".evidence.txt").write_text("changed", encoding="utf-8")
    elif mutation == "wrong_lock":
        other = tmp_path / "other.jsonl"
        other.write_bytes(development.read_bytes())
        lineage_lock = freeze_test_lineage(other)
    manifest.write_text("".join(json.dumps(r) + "\n" for r in rows), encoding="utf-8")
    lock_path.write_text(json.dumps(lock), encoding="utf-8")
    with pytest.raises(ValueError, match="development lineage"):
        freeze_external_test(manifest, development, policy, output, lock_path, lineage_lock)
    assert not output.exists()


@pytest.mark.parametrize("field,value", [
    ("completeness_status", "PENDING"), ("reviewer_type", "AI"),
    ("reviewer_id", "unknown"), ("reviewed_at", "2026-09-06T00:00:00"),
    ("adjudication_status", "PENDING"), ("covered_usage", ["TRAIN"]),
    ("lineage_sha256", "f" * 64),
])
def test_lineage_freeze_requires_complete_human_attestation(intake, tmp_path, field, value):
    _, dev = intake
    path = tmp_path / "lineage.jsonl"
    path.write_text(json.dumps(dev) + "\n", encoding="utf-8")
    freeze_test_lineage(path)
    review_path = path.with_suffix(".review.json")
    review = json.loads(review_path.read_text(encoding="utf-8"))
    review[field] = value
    review_path.write_text(json.dumps(review), encoding="utf-8")
    with pytest.raises(ValueError):
        freeze_development_lineage(path, review_path, tmp_path / "invalid-lock.json")
    assert not (tmp_path / "invalid-lock.json").exists()
