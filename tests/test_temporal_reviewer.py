"""Isolated UI tests. Temporary fixture decisions never touch real review data."""

import csv
import json

import pytest
from fastapi.testclient import TestClient

from tools.temporal_reviewer.app import HUMAN_FIELDS, create_app, read_queue
from training.validate_review1 import digest, payload_digest


@pytest.fixture
def workspace(tmp_path):
    media = tmp_path / "fixture.bin"
    media.write_bytes(b"ephemeral unit fixture; not real video or human evidence")
    payload = dict(
        source_page_url=None,
        license_or_terms_url=None,
        creator=None,
        asset_id=None,
        rights_recommendation="NEEDS_HUMAN_DECISION",
        rights_reason="Temporary test fixture only",
        evidence_available=False,
        evidence=[],
    )
    record = dict(
        record_type="rights",
        dataset_role="TEMPORAL_DEVELOPMENT",
        video_id="unit-fixture",
        video_path=str(media),
        video_sha256=digest(media),
        reviewer_type="AI",
        reviewer_origin="AI",
        review_stage="REVIEW1",
        reviewer_agent="CODEX",
        status="REVIEW1_PROPOSAL",
        requires_human_confirmation=True,
        adjudication_status="PENDING",
        human_approved=False,
        payload=payload,
    )
    source = tmp_path / "proposal.json"
    source.write_text(json.dumps(record), encoding="utf-8")
    queue = tmp_path / "queue.csv"
    rows = [
        dict(
            item_id=f"fixture_{i}",
            clip_id=f"clip_{i}",
            video_path=str(media),
            video_sha256=digest(media),
            record_path=str(source),
            record_sha256=digest(source),
            payload_sha256=payload_digest(payload),
            record_type="rights",
            proposed_item_json=json.dumps(payload),
            **dict.fromkeys(HUMAN_FIELDS, ""),
        )
        for i in range(2)
    ]
    with queue.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with TestClient(create_app(queue, tmp_path)) as client:
        state = client.get("/api/queue").json()
        body = dict(
            version=state["version"],
            reviewer_id="Nguyen-Van-An",
            reviewed_at="2026-09-07T10:30:00+07:00",
            notes="Isolated automated test fixture only; never human dataset evidence.",
            attested=True,
            items=[dict(item_id=r["item_id"], decision="APPROVE") for r in rows],
        )
        yield client, queue, source, media, state, body


def submit(workspace, body=None):
    client, _, _, _, state, default = workspace
    return client.post(
        "/api/decisions", json=body or default, headers={"X-Review-Token": state["token"]}
    )


def test_reads_are_side_effect_free_and_video_ranges_work(workspace):
    client, queue, source, media, _, _ = workspace
    before = (queue.read_bytes(), source.read_bytes())
    assert client.get("/").status_code == 200
    assert client.get("/assets/app.js").headers["content-type"].startswith("text/javascript")
    assert client.get("/assets/style.css").headers["content-type"].startswith("text/css")
    assert client.get("/api/export").status_code == 200
    video = client.get("/api/video/clip_0", headers={"Range": "bytes=0-7"})
    assert video.status_code == 206
    assert video.content == media.read_bytes()[:8]
    assert client.get("/api/video/missing").status_code == 404
    assert (queue.read_bytes(), source.read_bytes()) == before
    assert not (queue.parent / "human_decisions").exists()


def test_bulk_persists_receipts_preserving_ai_and_source(workspace):
    _, queue, source, _, _, _ = workspace
    original, source_bytes = queue.read_bytes(), source.read_bytes()
    result = submit(workspace)
    assert result.status_code == 200, result.text
    assert result.json()["saved"] == 2
    assert result.json()["governance_promotion"] is False
    rows, _, _, _ = read_queue(queue)
    for row in rows:
        receipt = queue.parent / row["REVIEW_EVIDENCE_PATH"]
        assert digest(receipt) == row["REVIEW_EVIDENCE_SHA256"]
        evidence = json.loads(receipt.read_text(encoding="utf-8"))
        assert evidence["record_sha256"] == digest(source)
        assert evidence["item"] == json.loads(row["proposed_item_json"])
        assert evidence["decision"] == "APPROVE"
        assert evidence["canonical_conversion"] is False
    assert source.read_bytes() == source_bytes
    assert json.loads(source_bytes)["human_approved"] is False
    snapshot = next((queue.parent / "human_decisions").glob("*/queue_before.csv"))
    assert snapshot.read_bytes() == original


@pytest.mark.parametrize(
    "field,value",
    [
        ("reviewer_id", "CODEX"),
        ("reviewer_id", "dummy-human"),
        ("reviewed_at", "2026-09-07T10:30:00"),
        ("reviewed_at", None),
        ("notes", ""),
        ("attested", False),
        ("version", "stale"),
    ],
)
def test_invalid_provenance_or_stale_request_never_writes(workspace, field, value):
    _, queue, _, _, _, body = workspace
    before = queue.read_bytes()
    body[field] = value
    assert submit(workspace).status_code in (400, 409)
    assert queue.read_bytes() == before
    assert not (queue.parent / "human_decisions").exists()


@pytest.mark.parametrize("target", ["source", "video", "item", "payload"])
def test_changed_source_bindings_reject_whole_batch(workspace, target):
    _, queue, source, media, state, body = workspace
    if target in ("source", "video"):
        path = source if target == "source" else media
        path.write_bytes(path.read_bytes() + b" ")
    else:
        rows, fields, _, _ = read_queue(queue)
        rows[1]["proposed_item_json" if target == "item" else "payload_sha256"] = (
            "{}" if target == "item" else "0" * 64
        )
        with queue.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        body["version"] = digest(queue)
    before = queue.read_bytes()
    assert submit(workspace).status_code == 409
    assert queue.read_bytes() == before


def test_no_overwrite_or_cross_origin_submission(workspace):
    client, queue, _, _, state, body = workspace
    assert client.post("/api/decisions", json=body).status_code == 403
    assert (
        client.post(
            "/api/decisions",
            json=body,
            headers={
                "X-Review-Token": state["token"],
                "Origin": "https://untrusted.invalid",
            },
        ).status_code
        == 403
    )
    assert submit(workspace).status_code == 200
    before = queue.read_bytes()
    body["version"] = digest(queue)
    assert submit(workspace).status_code == 409
    assert queue.read_bytes() == before


@pytest.mark.parametrize("decision", ["REJECT", "UNCERTAIN", "EDIT_AND_APPROVE"])
def test_alternative_decisions_remain_separate_from_proposal(workspace, decision):
    _, queue, source, _, _, body = workspace
    before = source.read_bytes()
    body["items"] = [dict(item_id="fixture_0", decision=decision)]
    if decision == "EDIT_AND_APPROVE":
        edited = json.loads(before)["payload"]
        edited["rights_reason"] = "Different reason in an isolated test fixture."
        body["items"][0]["edited_item"] = edited
    assert submit(workspace).status_code == 200
    rows, _, _, _ = read_queue(queue)
    assert rows[0]["HUMAN_DECISION"] == decision
    assert rows[1]["HUMAN_DECISION"] == ""
    assert source.read_bytes() == before


def test_bad_edit_does_not_write(workspace):
    _, queue, _, _, _, body = workspace
    before = queue.read_bytes()
    body["items"] = [dict(item_id="fixture_0", decision="EDIT_AND_APPROVE", edited_item={})]
    assert submit(workspace).status_code == 400
    assert queue.read_bytes() == before


@pytest.mark.parametrize("change", ["gap", "state", "time", "role", "valid"])
def test_sequence_edit_checks_entire_timeline(workspace, change):
    _, queue, source, _, _, body = workspace
    record = json.loads(source.read_text(encoding="utf-8"))
    record["record_type"] = "sequence"
    interval = dict(
        start_frame=0,
        end_frame=2,
        start_time=0,
        end_time=1,
        occupant_id_proposal="unit-person",
        occupant_role_proposal="unknown",
        visibility="partial",
        confidence=0.5,
        review_notes="Unit fixture only",
    )
    record["payload"] = dict(
        fps=2,
        frame_count=2,
        occupants=[
            dict(
                occupant_id_proposal="unit-person",
                role_proposal="unknown",
                inside_vehicle_proposal=None,
                evidence_basis="Unit fixture",
                confidence=0.5,
            )
        ],
        phone_intervals=[dict(interval, state="UNKNOWN")],
        seatbelt_intervals=[dict(interval, state="UNCERTAIN_OR_OCCLUDED")],
        context_intervals=[
            dict(
                interval,
                inside_vehicle=None,
                outside_vehicle_person=None,
                motorcycle_flag=None,
                conditions="unknown",
            )
        ],
    )
    source.write_text(json.dumps(record), encoding="utf-8")
    rows, fields, _, _ = read_queue(queue)
    item = record["payload"]["phone_intervals"][0]
    row = rows[0]
    row.update(
        record_type="sequence",
        record_sha256=digest(source),
        payload_sha256=payload_digest(record["payload"]),
        proposed_item_json=json.dumps(item),
    )
    with queue.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerow(row)
    body["version"] = digest(queue)
    edited = dict(item)
    if change == "gap":
        edited.update(start_frame=1, start_time=0.5)
    elif change == "state":
        edited["state"] = "BOGUS"
    elif change == "time":
        edited["end_time"] = 0.4
    elif change == "role":
        edited["occupant_role_proposal"] = "driver"
    else:
        edited["state"] = "NO_PHONE"
    body["items"] = [dict(item_id=row["item_id"], decision="EDIT_AND_APPROVE", edited_item=edited)]
    before = queue.read_bytes()
    response = submit(workspace)
    assert response.status_code == (200 if change == "valid" else 400), response.text
    if change != "valid":
        assert queue.read_bytes() == before
    assert json.loads(source.read_text(encoding="utf-8"))["payload"] == record["payload"]
