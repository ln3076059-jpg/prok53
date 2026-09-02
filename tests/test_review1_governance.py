from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image
from pydantic import ValidationError

from tools.annotation_reviewer import app as reviewer
from training.build_review1_contact_sheets import build as build_contact_sheets
from training.common import sha256_file
from training.materialize_review1_lanes import materialize
from training.run_review1_review import run


def _queue_and_visual(tmp_path: Path, *, visual_overrides: dict | None = None):
    datasets = tmp_path / "datasets"
    incoming = datasets / "incoming" / "source"
    manifests = datasets / "manifests"
    incoming.mkdir(parents=True)
    manifests.mkdir(parents=True)
    image = incoming / "sample-1.jpg"
    image.write_bytes(b"visually-inspected-image")
    digest = sha256_file(image)
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample-1",
                    "image_path": str(image),
                    "source_id": "source-1",
                    "source_asset_id": "asset-1",
                    "source_group_id": "group-1",
                    "review_status": "PENDING",
                    "annotations": [],
                }
            ]
        ),
        encoding="utf-8",
    )
    (manifests / "ingest_source.jsonl").write_text(
        json.dumps({"sample_id": "sample-1", "sha256": digest}) + "\n",
        encoding="utf-8",
    )
    visual = {
        "sample_id": "sample-1",
        "source_sha256": digest,
        "status": "REVIEW1_APPROVED_UNDER_ADMIN_DELEGATION",
        "decision_reason": "HARD_NEGATIVE_CONFIRMED",
        "reviewed_annotations": [],
        "review1_confidence": 0.99,
        "risk_flags": [],
        "visual_evidence": {
            "inspected": True,
            "method": "DIRECT_IMAGE_INSPECTION",
        },
        "vehicle_context_id": "cabin-1",
        "occupant_role": "driver",
        "video_id": "video-1",
        "vehicle_id": "vehicle-1",
        "person_id": "person-1",
        "camera_id": "camera-1",
        "conditions": ["daylight"],
    }
    visual.update(visual_overrides or {})
    visual_path = tmp_path / "visual.jsonl"
    visual_path.write_text(json.dumps(visual) + "\n", encoding="utf-8")
    return datasets, manifests, queue, visual_path, image, digest


def _run(tmp_path: Path, **kwargs):
    datasets, manifests, queue, visual, _, _ = _queue_and_visual(
        tmp_path, visual_overrides=kwargs.pop("visual_overrides", None)
    )
    output = tmp_path / "decisions.jsonl"
    report = run(
        queue,
        visual,
        output,
        tmp_path / "checkpoints.jsonl",
        tmp_path / "attention.json",
        manifests,
        datasets,
        batch_size=100,
        **kwargs,
    )
    return report, output, queue, datasets


def test_review1_records_ai_identity_delegation_and_append_only_resume(tmp_path):
    report, output, queue, datasets = _run(tmp_path)
    record = json.loads(output.read_text(encoding="utf-8"))

    assert report["total_reviewed"] == 1
    assert record["reviewer_id"] == "review1"
    assert record["reviewer_type"] == "AI"
    assert record["delegated_by"] == "admin"
    assert record["approval_authority_id"] == "admin"
    assert record["human_confirmation"] is False
    assert record["governance_eligible"] is False
    assert record["decision_id"]

    visual = tmp_path / "visual.jsonl"
    resumed = run(
        queue,
        visual,
        output,
        tmp_path / "checkpoints.jsonl",
        tmp_path / "attention.json",
        tmp_path / "datasets" / "manifests",
        datasets,
        resume=True,
        batch_size=100,
    )
    assert resumed["newly_reviewed"] == 0
    assert len(output.read_text(encoding="utf-8").splitlines()) == 1


def test_review1_cannot_emit_human_approval(tmp_path):
    with pytest.raises(ValueError, match="cannot emit status"):
        _run(tmp_path, visual_overrides={"status": "APPROVED"})


def test_review1_status_cannot_claim_human_reviewer_type():
    with pytest.raises(ValidationError, match="REVIEW1 statuses require reviewer_type=AI"):
        reviewer.Decision(
            sample_id="sample-1",
            reviewer_id="admin",
            reviewer_type="HUMAN",
            status="REVIEW1_UNCERTAIN",
            decision_reason="INSUFFICIENT_EVIDENCE",
            annotations=[],
        )


def test_review1_identity_cannot_claim_a_human_approval():
    with pytest.raises(ValidationError, match="cannot claim reviewer_type=HUMAN"):
        reviewer.Decision(
            sample_id="sample-1",
            reviewer_id="review1",
            reviewer_type="HUMAN",
            status="APPROVED_NEGATIVE",
            decision_reason="HARD_NEGATIVE_CONFIRMED",
            annotations=[],
            vehicle_context_id="cabin-1",
            occupant_role="driver",
            video_id="video-1",
            vehicle_id="vehicle-1",
            person_id="person-1",
            camera_id="camera-1",
            conditions=["daylight"],
        )


def test_review1_refuses_wrong_source_hash(tmp_path):
    with pytest.raises(ValueError, match="SOURCE_HASH_MISMATCH"):
        _run(tmp_path, visual_overrides={"source_sha256": "0" * 64})


def test_review1_rejects_invalid_bbox(tmp_path):
    with pytest.raises(ValueError):
        _run(
            tmp_path,
            visual_overrides={
                "status": "REVIEW1_CORRECTION_PROPOSAL",
                "decision_reason": "PHONE_MISSING_LABEL",
                "review1_confidence": 0.7,
                "risk_flags": ["MISSING_LABEL"],
                "reviewed_annotations": [{"class_id": 0, "yolo": [0.5, 0.5, 1.5, 0.2]}],
            },
        )


def test_uncertain_belt_cannot_become_unfastened(tmp_path):
    with pytest.raises(ValueError, match="cannot create an unfastened"):
        _run(
            tmp_path,
            visual_overrides={
                "status": "REVIEW1_UNCERTAIN",
                "decision_reason": "UNCERTAIN_OR_OCCLUDED",
                "review1_confidence": 0.6,
                "risk_flags": ["OCCLUDED"],
                "reviewed_annotations": [{"class_id": 2, "yolo": [0.5, 0.5, 0.2, 0.2]}],
            },
        )


def test_phone_missing_label_requires_correction_box(tmp_path):
    report, output, _, _ = _run(
        tmp_path,
        visual_overrides={
            "status": "REVIEW1_CORRECTION_PROPOSAL",
            "decision_reason": "PHONE_MISSING_LABEL",
            "review1_confidence": 0.8,
            "risk_flags": ["MISSING_LABEL"],
            "reviewed_annotations": [
                {
                    "class_id": 0,
                    "yolo": [0.5, 0.5, 0.2, 0.2],
                    "occupant_role": "UNCERTAIN",
                }
            ],
        },
    )
    record = json.loads(output.read_text(encoding="utf-8"))
    assert report["tier_counts"] == {"TIER_C": 1}
    assert record["reviewed_annotations"][0]["class_id"] == 0


def test_reviewer_api_keeps_review1_ai_and_admin_confirmation_separate(tmp_path, monkeypatch):
    datasets, _, queue, _, image, digest = _queue_and_visual(tmp_path)
    decisions = tmp_path / "api-decisions.jsonl"
    acknowledgements = tmp_path / "acknowledgements.jsonl"
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload[0]["sha256"] = digest
    payload[0]["proposal_review"] = {
        "status": "MODEL_ASSISTED_PROPOSAL_PENDING_APPROVAL",
        "decision": "PASS",
        "tier": "HIGH_CONFIDENCE",
    }
    queue.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(reviewer, "manifest_path", queue)
    monkeypatch.setattr(reviewer, "decisions_path", decisions)
    monkeypatch.setattr(reviewer, "acknowledgements_path", acknowledgements)
    monkeypatch.setattr(reviewer, "datasets_root", datasets)

    base = {
        "sample_id": "sample-1",
        "reviewer_id": "review1",
        "reviewer_type": "AI",
        "decision_reason": "HARD_NEGATIVE_CONFIRMED",
        "annotations": [],
        "delegated_by": "admin",
        "approval_authority_id": "admin",
        "review1_confidence": 0.99,
        "risk_flags": [],
        "visual_evidence": {
            "inspected": True,
            "method": "DIRECT_IMAGE_INSPECTION",
        },
        "vehicle_context_id": "cabin-1",
        "occupant_role": "driver",
        "video_id": "video-1",
        "vehicle_id": "vehicle-1",
        "person_id": "person-1",
        "camera_id": "camera-1",
        "conditions": ["daylight"],
    }
    with pytest.raises(ValidationError):
        reviewer.Decision(**base, status="APPROVED")
    with pytest.raises(ValidationError, match="requires a Review 1 correction proposal"):
        reviewer.Decision(
            **{**base, "decision_reason": "PHONE_MISSING_LABEL"},
            status="REVIEW1_UNCERTAIN",
        )

    reviewer.decision(
        reviewer.Decision(**base, status="REVIEW1_APPROVED_UNDER_ADMIN_DELEGATION")
    )
    queued = reviewer.queue()
    status_before_confirmation = reviewer.status()
    assert queued[0]["latest_review_decision"]["reviewer_id"] == "review1"
    assert status_before_confirmation["review1_decided"] == 1
    assert status_before_confirmation["human_confirmed"] == 0
    acknowledgement = reviewer.batch_acknowledgement(
        reviewer.BatchAcknowledgement(
            reviewer_id="admin",
            reviewer_type="HUMAN",
            sample_ids=["sample-1"],
        )
    )
    assert acknowledgement["saved"] is True
    ack = json.loads(acknowledgements.read_text(encoding="utf-8"))
    assert ack["human_confirmation"] is False
    assert ack["governance_eligible"] is False

    result = reviewer.admin_batch_confirm(
        reviewer.AdminBatchConfirmation(
            admin_actor_id="admin",
            reviewer_type="HUMAN",
            sample_ids=["sample-1"],
            confirmation_text="CONFIRM_REVIEW1_PROPOSALS_AS_ADMIN",
        )
    )
    history = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
    assert result["confirmed"] == 1
    assert history[0]["reviewer_type"] == "AI"
    assert history[1]["reviewer_type"] == "HUMAN"
    assert history[1]["reviewer_id"] == "admin"
    assert history[1]["confirmation_source"] == "ADMIN_BATCH_CONFIRM"
    assert history[1]["previous_status"] == "REVIEW1_APPROVED_UNDER_ADMIN_DELEGATION"
    assert reviewer.status()["human_confirmed"] == 1


def test_admin_batch_confirm_tier_b_with_metadata_and_auth(tmp_path, monkeypatch):
    import os

    from fastapi import HTTPException

    datasets, _, queue, _, image, digest = _queue_and_visual(tmp_path)
    decisions = tmp_path / "api-tierb-decisions.jsonl"
    payload = json.loads(queue.read_text(encoding="utf-8"))
    payload[0]["sha256"] = digest
    queue.write_text(json.dumps(payload), encoding="utf-8")

    monkeypatch.setattr(reviewer, "manifest_path", queue)
    monkeypatch.setattr(reviewer, "decisions_path", decisions)
    monkeypatch.setattr(reviewer, "datasets_root", datasets)
    monkeypatch.setattr(os, "environ", {**os.environ, "ADMIN_CONFIRMATION_SECRET": "secret123"})

    tier_b_proposal = {
        "sample_id": "sample-1",
        "reviewer_id": "review1",
        "reviewer_type": "AI",
        "decision_reason": "HARD_NEGATIVE_CONFIRMED",
        "annotations": [],
        "delegated_by": "admin",
        "approval_authority_id": "admin",
        "review1_confidence": 0.90,
        "risk_flags": [],
        "visual_evidence": {
            "inspected": True,
            "method": "DIRECT_IMAGE_INSPECTION",
        },
        "vehicle_context_id": "UNKNOWN",
        "occupant_role": "UNCERTAIN",
        "video_id": "UNKNOWN",
        "vehicle_id": "UNKNOWN",
        "person_id": "UNKNOWN",
        "camera_id": "UNKNOWN",
        "conditions": [],
    }

    reviewer.decision(
        reviewer.Decision(**tier_b_proposal, status="REVIEW1_ACCEPTED_PROPOSAL")
    )

    # Missing/wrong token should fail authentication with HTTP 401
    with pytest.raises(HTTPException) as exc_info:
        reviewer.admin_batch_confirm(
            reviewer.AdminBatchConfirmation(
                admin_actor_id="admin",
                reviewer_type="HUMAN",
                sample_ids=["sample-1"],
                confirmation_text="CONFIRM_REVIEW1_PROPOSALS_AS_ADMIN",
                admin_token="wrong_token",
            )
        )
    assert exc_info.value.status_code == 401

    # Successful admin confirmation supplying missing batch metadata overrides
    confirmed = reviewer.admin_batch_confirm(
        reviewer.AdminBatchConfirmation(
            admin_actor_id="admin",
            reviewer_type="HUMAN",
            sample_ids=["sample-1"],
            confirmation_text="CONFIRM_REVIEW1_PROPOSALS_AS_ADMIN",
            admin_token="secret123",
            vehicle_context_id="dms_context_01",
            video_id="video_dms_01",
            vehicle_id="vehicle_sedan_01",
            person_id="person_driver_01",
            camera_id="cam_front_01",
            conditions=["NORMAL_DAYLIGHT"],
            occupant_role="driver",
        )
    )
    assert confirmed["confirmed"] == 1
    records = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
    human_rec = records[1]
    assert human_rec["reviewer_type"] == "HUMAN"
    assert human_rec["status"] == "APPROVED_NEGATIVE"
    assert human_rec["vehicle_context_id"] == "dms_context_01"
    assert human_rec["occupant_role"] == "driver"
    assert human_rec["video_id"] == "video_dms_01"
    assert human_rec["conditions"] == ["NORMAL_DAYLIGHT"]
    assert human_rec["governance_eligible"] is True



def test_review1_correction_materializes_separate_lane_without_overwriting_source(tmp_path):
    image = tmp_path / "source.jpg"
    image.write_bytes(b"immutable-source-image")
    source_label = tmp_path / "source.txt"
    source_label.write_text("", encoding="utf-8")
    digest = sha256_file(image)
    records = [
        {
            "sample_id": "sample-1",
            "status": "PENDING",
            "ingested_path": str(image),
            "source_group_id": "source:group-1",
            "sha256": digest,
        }
    ]
    correction = {
        "sample_id": "sample-1",
        "reviewer_id": "review1",
        "reviewer_type": "AI",
        "status": "REVIEW1_CORRECTION_PROPOSAL",
        "new_status": "REVIEW1_CORRECTION_PROPOSAL",
        "delegated_by": "admin",
        "approval_authority_id": "admin",
        "source_sha256": digest,
        "decision_id": "review1-decision",
        "reviewed_annotations": [
            {
                "class_id": 0,
                "yolo": [0.5, 0.5, 0.2, 0.2],
                "occupant_role": "UNCERTAIN",
            }
        ],
    }

    bootstrap, governed = materialize(
        records,
        [correction],
        tmp_path / "bootstrap",
        tmp_path / "governed",
    )

    assert bootstrap[0]["governance_status"] == "NOT_GOVERNED"
    assert bootstrap[0]["human_approval_status"] == "NOT_HUMAN_APPROVED"
    assert governed == []
    assert image.read_bytes() == b"immutable-source-image"
    assert source_label.read_text(encoding="utf-8") == ""
    assert Path(bootstrap[0]["label_path"]).read_text(encoding="utf-8").startswith("0 ")

    resumed_bootstrap, resumed_governed = materialize(
        records,
        [correction],
        tmp_path / "bootstrap",
        tmp_path / "governed",
    )
    assert resumed_bootstrap == bootstrap
    assert resumed_governed == []


def test_reviewer_ui_names_review1_and_requires_explicit_admin_confirmation():
    html = Path("tools/annotation_reviewer/index.html").read_text(encoding="utf-8")

    assert "Review 1 — độ tin cậy cao" in html
    assert "Bạn đang xác nhận các đề xuất do Review 1 rà soát với tư cách admin." in html
    assert "CONFIRM_REVIEW1_PROPOSALS_AS_ADMIN" in html
    assert 'id="adminConfirm"' in html


def test_contact_sheet_offset_selects_the_next_deterministic_batch(tmp_path):
    queue = []
    datasets = tmp_path / "datasets"
    datasets.mkdir()
    for index in range(3):
        image = datasets / f"sample-{index}.jpg"
        Image.new("RGB", (20, 20), color=(index * 20, 0, 0)).save(image)
        queue.append(
            {
                "sample_id": f"sample-{index}",
                "image_path": str(image),
                "annotations": [],
                "proposal_review": {"reason": ["target"]},
            }
        )
    manifest = tmp_path / "queue.json"
    manifest.write_text(json.dumps(queue), encoding="utf-8")

    report = build_contact_sheets(
        manifest,
        tmp_path / "sheets",
        reason="target",
        offset=1,
        limit=1,
        per_page=1,
        columns=1,
        tile_width=100,
        tile_height=100,
        datasets_root=datasets,
    )
    selected = json.loads((tmp_path / "sheets" / "page_001.json").read_text())

    assert report["selection_offset"] == 1
    assert selected[0]["sample_id"] == "sample-1"
