from __future__ import annotations

import json
from pathlib import Path

from training.dmd.prepare_v31_component_truth import build_manifest, validate_manifest


def test_component_truth_is_pending_until_human_review(tmp_path: Path) -> None:
    sheet = tmp_path / "case.jpg"
    sheet.write_bytes(b"image")
    source = tmp_path / "audit.json"
    source.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "subject_id": "gC-14",
                        "case_id": "FN-01",
                        "kind": "FN",
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                        "candidate_count": 1,
                        "sampled_timestamps": [1.5],
                        "contact_sheet": str(sheet),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = build_manifest(source)
    result = validate_manifest(payload)
    assert payload["status"] == "PENDING_HUMAN_REVIEW"
    assert result["training_eligible"] is False
    assert result["errors"] == []


def test_completed_component_truth_requires_resolved_review(tmp_path: Path) -> None:
    sheet = tmp_path / "case.jpg"
    sheet.write_bytes(b"image")
    source = tmp_path / "audit.json"
    source.write_text(
        json.dumps(
            {
                "cases": [
                    {
                        "subject_id": "gC-14",
                        "case_id": "FP-01",
                        "kind": "FP",
                        "start_seconds": 1.0,
                        "end_seconds": 2.0,
                        "candidate_count": 1,
                        "sampled_timestamps": [1.5],
                        "contact_sheet": str(sheet),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    payload = build_manifest(source)
    payload["cases"][0]["review"] = {
        "status": "REVIEWED",
        "reviewer_id": "reviewer-1",
        "reviewed_at": "2026-10-23T10:00:00+07:00",
        "phone_state": "ABSENT",
        "driver_association": "UNKNOWN",
        "proposal_quality": "WRONG_OBJECT",
        "primary_failure_layer": "PROPOSAL_LOCALIZATION",
        "confidence_1_to_3": 3,
        "notes": "",
    }
    result = validate_manifest(payload)
    assert result["status"] == "PASS_COMPLETE"
    assert result["training_eligible"] is True
