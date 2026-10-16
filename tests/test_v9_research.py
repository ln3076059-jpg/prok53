from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from backend.schemas import PhoneResearchReviewRequest
from backend.services.research_review import (
    ResearchReviewConflict,
    ResearchReviewError,
    list_phone_reviews,
    phone_review_detail,
    save_phone_review,
)
from training.dmd.v9_diagnostics import (
    assert_development_subject,
    classify_fn_bottleneck,
    summarize_detection_row,
)
from training.dmd.v9_phone_proposal_mining import intersection_over_union


def test_missing_phone_box_is_explicit_and_blocks_geometry():
    summary = summarize_detection_row(
        {
            "samples": [
                {"decode_status": "OK", "phone_proposals": []},
                {"decode_status": "OK", "phone_proposals": []},
            ]
        }
    )
    assert summary["phone_box_status"] == "MISSING_ALL_SAMPLED_FRAMES"
    assert summary["occupant_id_status"] == "BLOCKED_NO_PHONE_BOX"
    assert summary["geometry_status"] == "BLOCKED_NO_PHONE_BOX"
    assert classify_fn_bottleneck(summary, 0.95) == "PHONE_DETECTOR_NO_PROPOSAL"


def test_roi_level_phone_box_does_not_claim_occupant_identity():
    summary = summarize_detection_row(
        {
            "samples": [
                {
                    "decode_status": "OK",
                    "phone_proposals": [
                        {
                            "confidence": 0.42,
                            "occupant_association": "DRIVER_ROI_LEVEL_ONLY_NO_TRACK_ID",
                        }
                    ],
                },
                {"decode_status": "OK", "phone_proposals": []},
            ]
        }
    )
    assert summary["phone_box_status"] == "AVAILABLE_ON_SUBSET"
    assert summary["occupant_id_status"] == "MISSING_TRACK_ID"
    assert summary["geometry_status"] == "UNVERIFIED_ROI_LEVEL_ASSOCIATION"
    assert classify_fn_bottleneck(summary, 0.55) == "PHONE_BOX_PRESENT_ACTION_UNDER_GATE"


@pytest.mark.parametrize("subject", ["gZ-37", "gE-28", "unknown"])
def test_v9_rejects_protected_or_unregistered_subjects(subject: str):
    with pytest.raises(ValueError, match="not registered"):
        assert_development_subject(subject)


def test_v9_source_artifacts_share_exact_thirteen_fn_cases():
    evidence = json.loads(Path("reports/ai85_plus/evidence_diagnostics.json").read_text())
    detector = json.loads(Path("reports/ai85_next/detector_trace.json").read_text())
    manifest = json.loads(Path("reports/ai85_upgrade/fn_frame_manifest.json").read_text())
    expected = {item["case_id"] for item in evidence["false_negative_cases"]}
    assert len(expected) == 13
    assert expected == {item["case_id"] for item in detector["rows"]}
    assert expected == {item["case_id"] for item in manifest}
    assert not any(name in json.dumps(evidence) for name in ("gZ-37", "gE-28"))


def test_phone_proposal_iou_supports_review_without_false_overlap():
    assert intersection_over_union([0, 0, 10, 10], [5, 5, 15, 15]) == pytest.approx(25 / 175)
    assert intersection_over_union([0, 0, 10, 10], [11, 11, 20, 20]) == 0.0
    assert intersection_over_union([1, 1, 1, 1], [1, 1, 1, 1]) == 0.0


def _research_review_fixture(tmp_path: Path) -> tuple[Path, Path]:
    contact = tmp_path / "phone_proposal_review" / "case-00.jpg"
    contact.parent.mkdir()
    contact.write_bytes(b"review-image")
    rows = []
    for index in range(13):
        case_id = f"case-{index:02d}"
        sheet = contact
        if index:
            sheet = contact.with_name(f"{case_id}.jpg")
            sheet.write_bytes(f"review-image-{index}".encode())
        rows.append(
            {
                "case_id": case_id,
                "subject_id": "gB-9",
                "specialist_proposals": 1,
                "auxiliary_proposals": 0,
                "cross_model_agreement": False,
                "auxiliary_review_candidate": False,
                "contact_sheet": str(sheet),
                "samples": [
                    {
                        "frame_index": index,
                        "timestamp_seconds": float(index),
                        "proposals": [
                            {
                                "proposal_id": f"f{index}-S1",
                                "source": "specialist",
                                "confidence": 0.5,
                                "geometry_context": "HANDHELD_USE",
                                "hand_proximity": 0.8,
                                "face_proximity": 0.1,
                                "nearest_cross_model_iou": 0.0,
                                "xyxy_in_driver_roi": [1, 2, 3, 4],
                            }
                        ],
                    }
                ],
            }
        )
    source = tmp_path / "phone_proposal_mining.json"
    source.write_text(json.dumps({"scope": "review-only fixture", "rows": rows}), encoding="utf-8")
    return source, tmp_path / "decisions"


def test_research_review_writes_versioned_human_receipts(tmp_path: Path):
    source, decisions = _research_review_fixture(tmp_path)
    payload = {
        "phone_visibility": "PHONE_VISIBLE",
        "phone_use_assessment": "PHONE_USE_CONFIRMED",
        "selected_proposal_id": "f0-S1",
        "notes": "Phone is visible in the driver's hand.",
        "expected_revision": None,
    }

    first = save_phone_review(source, decisions, "case-00", payload, "reviewer-1")
    detail = phone_review_detail(source, decisions, "case-00")
    queue = list_phone_reviews(source, decisions)

    assert first["reviewer_type"] == "HUMAN"
    assert first["selected_proposal"]["proposal_id"] == "f0-S1"
    assert first["training_admission"].startswith("NOT_AUTOMATIC")
    assert detail["decision"]["revision"] == first["revision"]
    assert queue["summary"] == {"total": 13, "reviewed": 1, "pending": 12, "recheck": 0}
    assert len(list((decisions / "history" / "case-00").glob("*.json"))) == 1

    with pytest.raises(ResearchReviewConflict, match="another session"):
        save_phone_review(source, decisions, "case-00", payload, "reviewer-2")

    revised = {
        **payload,
        "notes": "Second human pass confirms the same visible phone box.",
        "expected_revision": first["revision"],
    }
    second = save_phone_review(source, decisions, "case-00", revised, "reviewer-2")
    assert second["previous_revision"] == first["revision"]
    assert len(list((decisions / "history" / "case-00").glob("*.json"))) == 2


def test_research_review_exposes_ai_suggestion_but_receipt_remains_human(tmp_path: Path):
    source, decisions = _research_review_fixture(tmp_path)
    source_rows = json.loads(source.read_text(encoding="utf-8"))["rows"]
    suggestions = []
    for row in source_rows:
        proposal_id = row["samples"][0]["proposals"][0]["proposal_id"]
        suggestion = {
            "case_id": row["case_id"],
            "reviewer_type": "AI",
            "reviewer_id": "openai-codex",
            "status": "AI_PRE_REVIEW",
            "phone_visibility": "PHONE_VISIBLE",
            "phone_use_assessment": "PHONE_USE_CONFIRMED",
            "selected_proposal_id": proposal_id,
            "confidence": "HIGH",
            "rationale": "The phone and proposed box are visible in the contact sheet.",
        }
        canonical = json.dumps(
            suggestion, sort_keys=True, separators=(",", ":"), ensure_ascii=False
        ).encode()
        suggestion["revision"] = hashlib.sha256(canonical).hexdigest()
        suggestions.append(suggestion)
    (source.parent / "codex_phone_pre_reviews.json").write_text(
        json.dumps({"rows": suggestions}), encoding="utf-8"
    )

    detail = phone_review_detail(source, decisions, "case-00")
    suggestion = detail["pre_review"]
    assert suggestion["reviewer_type"] == "AI"
    assert detail["pre_review_available"] is True

    receipt = save_phone_review(
        source,
        decisions,
        "case-00",
        {
            "phone_visibility": suggestion["phone_visibility"],
            "phone_use_assessment": suggestion["phone_use_assessment"],
            "selected_proposal_id": suggestion["selected_proposal_id"],
            "notes": suggestion["rationale"],
            "expected_revision": None,
            "pre_review_revision": suggestion["revision"],
        },
        "human-reviewer-1",
    )

    assert receipt["reviewer_type"] == "HUMAN"
    assert receipt["reviewer_id"] == "human-reviewer-1"
    assert receipt["review_assistance"] == {
        "type": "AI_PRE_REVIEW",
        "reviewer_id": "openai-codex",
        "revision": suggestion["revision"],
        "accepted_without_changes": True,
    }

    suggestions[0]["rationale"] = "A second AI pass recommends that a human check this case again."
    suggestions[0].pop("revision")
    canonical = json.dumps(
        suggestions[0], sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode()
    suggestions[0]["revision"] = hashlib.sha256(canonical).hexdigest()
    (source.parent / "codex_phone_pre_reviews.json").write_text(
        json.dumps({"rows": suggestions}), encoding="utf-8"
    )
    queue = list_phone_reviews(source, decisions)
    assert queue["summary"] == {"total": 13, "reviewed": 0, "pending": 12, "recheck": 1}
    assert queue["cases"][0]["review_status"] == "RECHECK"
    assert phone_review_detail(source, decisions, "case-00")[
        "pre_review_changed_since_decision"
    ] is True


def test_research_review_rejects_foreign_proposal_and_inconsistent_decision(tmp_path: Path):
    source, decisions = _research_review_fixture(tmp_path)
    with pytest.raises(ResearchReviewError, match="does not belong"):
        save_phone_review(
            source,
            decisions,
            "case-00",
            {
                "phone_visibility": "PHONE_VISIBLE",
                "phone_use_assessment": "UNRESOLVED",
                "selected_proposal_id": "f99-S1",
                "notes": "Proposal belongs to a different review case.",
                "expected_revision": None,
            },
            "reviewer-1",
        )
    with pytest.raises(ValueError, match="No visible phone"):
        PhoneResearchReviewRequest(
            phone_visibility="NO_PHONE_VISIBLE",
            phone_use_assessment="PHONE_USE_CONFIRMED",
            selected_proposal_id="f0-S1",
            notes="This decision is intentionally inconsistent.",
        )
