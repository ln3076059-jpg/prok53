"""Final-review UI tests use disposable fixture files, never real human evidence."""

import copy
import csv
import json
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from tools.temporal_reviewer.app import create_app
from tools.temporal_reviewer.final_review import CLIPS
from training.validate_review1 import digest, payload_digest


def save(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj), encoding="utf-8")


@pytest.fixture
def ws(tmp_path):
    base = tmp_path / "local"
    queue = base / "review1/HUMAN_APPROVAL_QUEUE.csv"
    queue.parent.mkdir(parents=True)
    all_rows, candidates = [], {}
    for clip in CLIPS:
        video = base / f"{clip}.bin"
        video.write_bytes(b"DISPOSABLE UNIT FIXTURE, NOT REAL MEDIA " + clip.encode())
        payload = {
            "rights": {"project_use_review": "PENDING"},
            "physical_lineage": {"lineage_status": "NOT_PROVABLE"},
            "identity": {"occupants": []},
            "sequence": {
                "phone_intervals": [{"start_frame": 0, "end_frame": 1}],
                "seatbelt_intervals": [{"start_frame": 0, "end_frame": 1}],
                "context_intervals": [{"start_frame": 0, "end_frame": 1}],
            },
        }
        c = {
            "clip_id": clip,
            "dataset_role": "TEMPORAL_DEVELOPMENT",
            "video_path": str(video),
            "video_sha256": digest(video),
            "assembled_payload": payload,
            "assembled_payload_sha256": payload_digest(payload),
            "assembled_record_payload_sha256": {k: payload_digest(v) for k, v in payload.items()},
            "source_review1_records": {},
            "source_review1_record_sha256": {},
            "approved_item_receipts": [],
        }
        for kind, content in payload.items():
            source = base / "review1" / f"{clip}_{kind}.json"
            rec = {
                "video_sha256": c["video_sha256"],
                "status": "REVIEW1_PROPOSAL",
                "human_approved": False,
                "payload": content,
            }
            save(source, rec)
            c["source_review1_records"][kind] = {
                "path": str(source),
                "sha256": digest(source),
                "payload_sha256": payload_digest(content),
            }
            c["source_review1_record_sha256"][str(source)] = digest(source)
            parts = (
                [(f"/{kind}", content)]
                if kind != "sequence"
                else [(f"/sequence/{k}/0", v[0]) for k, v in content.items()]
            )
            for pointer, value in parts:
                item_id = clip + pointer.replace("/", "_")
                ep = base / "review1" / f"{item_id}_receipt.json"
                receipt = {
                    "item_id": item_id,
                    "item": value,
                    "decision": "APPROVE",
                    "video_sha256": c["video_sha256"],
                    "record_sha256": digest(source),
                    "payload_sha256": payload_digest(content),
                    "reviewer_id": "admin",
                    "reviewed_at": "2026-09-07T07:00:00+07:00",
                    "review_notes": "Isolated unit fixture.",
                }
                save(ep, receipt)
                row = {
                    "item_id": item_id,
                    "clip_id": clip,
                    "record_path": str(source),
                    "video_sha256": c["video_sha256"],
                    "record_sha256": digest(source),
                    "payload_sha256": payload_digest(content),
                    "REVIEWER_ID": "admin",
                    "REVIEWED_AT": receipt["reviewed_at"],
                    "HUMAN_DECISION": "APPROVE",
                    "HUMAN_NOTES": receipt["review_notes"],
                    "EDITED_ITEM_JSON": "",
                    "proposed_item_json": json.dumps(value),
                    "REVIEW_EVIDENCE_PATH": str(ep),
                    "REVIEW_EVIDENCE_SHA256": digest(ep),
                }
                all_rows.append(row)
                c["approved_item_receipts"].append(
                    {
                        "item_id": item_id,
                        "path": str(ep),
                        "sha256": digest(ep),
                        "source_record_path": str(source),
                        "source_record_sha256": digest(source),
                        "source_payload_sha256": payload_digest(content),
                        "reviewer_id": "admin",
                        "reviewed_at": receipt["reviewed_at"],
                        "decision": "APPROVE",
                        "assembled_item_pointer": pointer,
                        "approved_item": value,
                        "approved_item_sha256": payload_digest(value),
                    }
                )
        candidates[clip] = c
    with queue.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    for clip, c in candidates.items():
        c["source_queue"] = {"path": str(queue), "sha256": digest(queue)}
        cp = base / "final_payload_candidates" / f"{clip}_final_payload_candidate.json"
        save(cp, c)
        template = {
            "clip_id": clip,
            "candidate_path": str(cp),
            "candidate_sha256": digest(cp),
            "video_sha256": c["video_sha256"],
            "final_payload_sha256": c["assembled_payload_sha256"],
            "record_type": "sequence",
            "payload_sha256": c["assembled_record_payload_sha256"]["sequence"],
            "record_payload_sha256": c["assembled_record_payload_sha256"],
            "reviewer_id": None,
            "reviewed_at": None,
            "decision": None,
            "review_notes": None,
        }
        save(base / "final_payload_review" / f"{clip}_final_payload_receipt.json", template)
    with TestClient(create_app(queue, tmp_path)) as client:
        state = client.get("/api/final-review").json()
        assert all(c["error"] is None for c in state["clips"])
        yield client, base, state


def body(ws, kind="payload", clip="C01"):
    state = ws[0].get("/api/final-review").json()
    c = next(c for c in state["clips"] if c["clip_id"] == clip)
    return {
        "candidate_version": c["version"],
        "document_version": c["states"][kind]["version"],
        "reviewer_id": "admin",
        "reviewed_at": "2026-09-07T09:00:00+07:00",
        "review_notes": "Isolated automated test only; not human dataset evidence.",
        "attested": True,
        "decision": "APPROVE",
    }


def post(ws, kind, value, clip="C01"):
    return ws[0].post(
        f"/api/final-review/{clip}/{kind}", json=value, headers={"X-Review-Token": ws[2]["token"]}
    )


def upload(ws, kind="rights", clip="C01", content=b"Isolated documentary bytes for unit test"):
    return ws[0].post(
        f"/api/final-review/{clip}/evidence/{kind}",
        content=content,
        headers={"X-Review-Token": ws[2]["token"], "X-File-Name": "fixture.txt"},
    )


def rights(ws, decision="ACCEPT"):
    result = body(ws, "rights")
    result.update(
        decision=decision,
        source_page_url="https://example.invalid/source",
        license_or_terms_url="https://example.invalid/terms",
        project_use="Unit fixture only",
        evidence_confirmed=True,
        upload_ids=[],
    )
    return result


def test_read_only_overview_and_assets(ws):
    client, base, _ = ws
    before = {str(p): digest(p) for p in base.rglob("*") if p.is_file()}
    page = client.get("/final")
    script = client.get("/assets/final.js")
    assert page.status_code == 200
    assert script.headers["content-type"].startswith("text/javascript")
    assert "REVIEW1 ĐÃ HOÀN TẤT" in page.text
    assert "Xác nhận 4 bản tổng hợp" in page.text
    assert "Lưu toàn bộ xác nhận cho 4 clip" in page.text
    assert "Codex" not in page.text + script.text
    assert client.get("/assets/final.css").headers["content-type"].startswith("text/css")
    assert client.get("/api/final-review/C01/download/candidate").status_code == 200
    assert {str(p): digest(p) for p in base.rglob("*") if p.is_file()} == before
    assert not (base / "human_review_return").exists()


def test_full_payload_confirm_preserves_candidates_and_item_receipts(ws):
    _, base, _ = ws
    originals = {
        str(p): digest(p)
        for folder in ["review1", "final_payload_candidates"]
        for p in (base / folder).rglob("*")
        if p.is_file()
    }
    value = body(ws)
    response = post(ws, "payload", value)
    assert response.status_code == 200, response.text
    receipt = json.loads((base / "final_payload_review/C01_final_payload_receipt.json").read_text())
    assert receipt["decision"] == "APPROVE" and receipt["reviewer_id"] == "admin"
    assert receipt["governance_promotion"] is False
    assert receipt["payload_sha256"] != receipt["final_payload_sha256"]
    assert all(digest(p) == h for p, h in originals.items())
    assert post(ws, "payload", body(ws)).status_code == 409
    assert (
        len(list((base / "human_review_return/governance_ui/history").glob("*/before.json"))) == 1
    )


@pytest.mark.parametrize(
    "field,value",
    [
        ("reviewer_id", "CODEX"),
        ("reviewer_id", None),
        ("reviewed_at", "2026-09-07T09:00:00"),
        ("review_notes", "approve"),
        ("attested", False),
        ("candidate_version", "stale"),
        ("document_version", "stale"),
        ("decision", "EDIT_AND_APPROVE"),
    ],
)
def test_invalid_confirmation_does_not_write(ws, field, value):
    b = body(ws)
    b[field] = value
    before = digest(ws[1] / "final_payload_review/C01_final_payload_receipt.json")
    assert post(ws, "payload", b).status_code in (400, 409)
    assert digest(ws[1] / "final_payload_review/C01_final_payload_receipt.json") == before
    assert not (ws[1] / "human_review_return").exists()


@pytest.mark.parametrize(
    "target", ["video", "queue", "source", "item_receipt", "candidate", "template"]
)
def test_changed_bindings_reject_confirmation(ws, target):
    b = body(ws)
    base = ws[1]
    cp = base / "final_payload_candidates/C01_final_payload_candidate.json"
    c = json.loads(cp.read_text())
    p = {
        "video": base / "C01.bin",
        "queue": base / "review1/HUMAN_APPROVAL_QUEUE.csv",
        "source": Path(c["source_review1_records"]["rights"]["path"]),
        "item_receipt": Path(c["approved_item_receipts"][0]["path"]),
        "candidate": cp,
        "template": base / "final_payload_review/C01_final_payload_receipt.json",
    }[target]
    if target == "template":
        value = json.loads(p.read_text())
        value["final_payload_sha256"] = "0" * 64
        save(p, value)
    else:
        p.write_bytes(p.read_bytes() + b" ")
    assert post(ws, "payload", b).status_code == 409
    assert not (base / "human_review_return").exists()


def test_partial_item_coverage_rejected_even_if_candidate_rehashed(ws):
    base = ws[1]
    cp = base / "final_payload_candidates/C01_final_payload_candidate.json"
    c = json.loads(cp.read_text())
    c["approved_item_receipts"].pop()
    save(cp, c)
    rp = base / "final_payload_review/C01_final_payload_receipt.json"
    r = json.loads(rp.read_text())
    r["candidate_sha256"] = digest(cp)
    save(rp, r)
    assert post(ws, "payload", body(ws)).status_code == 409


def test_rights_requires_document_and_attestation_then_records_human_decision(ws):
    b = rights(ws)
    assert post(ws, "rights", b).status_code == 400
    evidence = upload(ws).json()
    b["upload_ids"] = [evidence["upload_id"]]
    b["evidence_confirmed"] = False
    assert post(ws, "rights", b).status_code == 400
    b["evidence_confirmed"] = True
    response = post(ws, "rights", b)
    assert response.status_code == 200, response.text
    doc = json.loads((ws[1] / "human_review_return/governance_ui/C01_rights.json").read_text())
    assert doc["project_use_review"] == "HUMAN_APPROVED"
    assert doc["submission_status"] == "PENDING_EVIDENCE_VERIFICATION"
    assert doc["governance_promotion"] is False
    evidence_file = ws[1].parent / doc["evidence"][0]["path"]
    assert digest(evidence_file) == doc["evidence"][0]["sha256"]
    response = ws[0].get("/api/final-review/evidence/" + evidence["upload_id"])
    assert response.headers["content-disposition"].startswith("attachment")


@pytest.mark.parametrize("kind,clip", [("rights", "C07"), ("lineage", "C01")])
def test_foreign_evidence_cannot_be_relabelled(ws, kind, clip):
    b = rights(ws)
    b["upload_ids"] = [upload(ws, kind, clip).json()["upload_id"]]
    assert post(ws, "rights", b).status_code == 400


def test_tampered_upload_rejected(ws):
    b = rights(ws)
    info = upload(ws).json()
    b["upload_ids"] = [info["upload_id"]]
    (
        ws[1] / "human_review_return/governance_ui/uploads" / info["upload_id"] / "content"
    ).write_bytes(b"changed")
    assert post(ws, "rights", b).status_code == 409


def test_not_provable_needs_no_fake_ids_or_documents(ws):
    b = body(ws, "lineage")
    b.update(decision="NOT_PROVABLE", physical_ids={}, upload_ids=[])
    assert post(ws, "lineage", b).status_code == 200
    doc = json.loads((ws[1] / "human_review_return/governance_ui/C01_lineage.json").read_text())
    assert doc["physical_ids"] is None and doc["physical_lineage_status"] == "NOT_PROVABLE"
    newer = body(ws, "lineage")
    newer.update(decision="NOT_PROVABLE", physical_ids={"source_id": "guess"})
    assert post(ws, "lineage", newer).status_code == 400


def test_multi_vehicle_submission_requires_mapping_and_is_only_pending_verification(ws):
    b = body(ws, "lineage")
    info = upload(ws, "lineage").json()
    b.update(
        decision="SUBMIT_FOR_VERIFICATION",
        upload_ids=[info["upload_id"]],
        evidence_confirmed=True,
        multiple_vehicles=True,
        physical_ids={
            "source_id": "fixture-source",
            "camera_id": "fixture-camera",
            "capture_session_id": "fixture-session",
            "physical_vehicle_group_id": "fixture-car-a",
            "person_group_ids": ["fixture-person"],
            "vehicle_physical_groups": {},
        },
    )
    assert post(ws, "lineage", b).status_code == 400
    b["physical_ids"]["vehicle_physical_groups"] = {
        "car-a": "fixture-car-a",
        "car-b": "fixture-car-b",
    }
    assert post(ws, "lineage", b).status_code == 200
    doc = json.loads((ws[1] / "human_review_return/governance_ui/C01_lineage.json").read_text())
    assert doc["physical_lineage_status"] == "PENDING_VERIFICATION"
    assert doc["governance_promotion"] is False


def test_rights_revision_preserves_previous_and_rejects_stale_client(ws):
    b = rights(ws, "NEEDS_HUMAN_DECISION")
    assert post(ws, "rights", b).status_code == 200
    previous = (ws[1] / "human_review_return/governance_ui/C01_rights.json").read_bytes()
    assert post(ws, "rights", b).status_code == 409
    updated = rights(ws, "NEEDS_HUMAN_DECISION")
    updated["review_notes"] += " Another documented note."
    assert post(ws, "rights", updated).status_code == 200
    assert any(
        p.read_bytes() == previous
        for p in (ws[1] / "human_review_return/governance_ui/history").glob("*/before.json")
    )


def test_upload_limits_paths_and_csrf(ws, monkeypatch):
    client = ws[0]
    route = "/api/final-review/C01/evidence/rights"
    assert client.post(route, content=b"bytes").status_code == 403
    headers = {"X-Review-Token": ws[2]["token"], "X-File-Name": "../escape.txt"}
    assert client.post(route, content=b"bytes", headers=headers).status_code == 400
    headers["X-File-Name"] = "fixture.txt"
    headers["Origin"] = "https://outside.invalid"
    assert client.post(route, content=b"bytes", headers=headers).status_code == 403
    assert upload(ws, content=b"").status_code == 400
    monkeypatch.setattr("tools.temporal_reviewer.final_review.MAX_UPLOAD", 4)
    assert upload(ws, content=b"oversized").status_code == 413


def test_displayed_versions_reject_another_tabs_new_submission(ws):
    first = rights(ws, "NEEDS_HUMAN_DECISION")
    second = copy.deepcopy(first)
    assert post(ws, "rights", first).status_code == 200
    second["review_notes"] = "Stale browser with a different decision note."
    assert post(ws, "rights", second).status_code == 409
