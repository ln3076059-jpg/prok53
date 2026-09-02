from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

from tools.annotation_reviewer import app as reviewer


def test_reviewer_interface_is_vietnamese_without_changing_governance_values():
    page = Path("tools/annotation_reviewer/index.html").read_text(encoding="utf-8")

    assert '<html lang="vi">' in page
    assert "Bằng chứng do Review 1 hỗ trợ" in page
    assert "Quyết định của người rà soát" in page
    assert "Phê duyệt mẫu âm" in page
    assert "Đánh dấu không chắc chắn" in page
    assert 'data-status="APPROVED_NEGATIVE"' in page
    assert 'reviewer_type: "HUMAN"' in page


def test_reviewer_documented_module_launcher_imports_from_repository_root():
    result = subprocess.run(
        [sys.executable, "-m", "tools.annotation_reviewer.app", "--help"],
        check=False,
        capture_output=True,
        text=True,
        timeout=30,
    )

    assert result.returncode == 0, result.stderr
    assert "--manifest" in result.stdout


def test_reviewer_appends_transition_reason_source_and_stable_box_id(tmp_path, monkeypatch):
    manifest = tmp_path / "queue.json"
    decisions = tmp_path / "decisions.jsonl"
    image = tmp_path / "sample-1.jpg"
    image.write_bytes(b"review-image")
    manifest.write_text(
        json.dumps(
            [
                {
                    "sample_id": "sample-1",
                    "source_id": "source-1",
                    "source_asset_id": "asset-1",
                    "source_group_id": "group-1",
                    "sha256": reviewer.sha256_file(image),
                    "image_path": str(image),
                }
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(reviewer, "manifest_path", manifest)
    monkeypatch.setattr(reviewer, "decisions_path", decisions)
    monkeypatch.setattr(reviewer, "datasets_root", tmp_path)

    first = reviewer.Decision(
        sample_id="sample-1",
        reviewer_id="reviewer@example.org",
        reviewer_type="HUMAN",
        status="UNCERTAIN",
        decision_reason="INSUFFICIENT_EVIDENCE",
        annotations=[
            reviewer.Annotation(
                class_id=0,
                yolo=(0.5, 0.5, 0.2, 0.2),
                occupant_role="UNCERTAIN",
            )
        ],
    )
    reviewer.decision(first)
    second = reviewer.Decision(
        sample_id="sample-1",
        reviewer_id="reviewer@example.org",
        reviewer_type="HUMAN",
        status="REJECTED",
        decision_reason="PHONE_FALSE_POSITIVE",
        annotations=[],
    )
    reviewer.decision(second)
    duplicate = reviewer.decision(second)

    records = [json.loads(line) for line in decisions.read_text(encoding="utf-8").splitlines()]
    assert len(records) == 2
    assert records[0]["previous_status"] == "PENDING"
    assert records[0]["new_status"] == "UNCERTAIN"
    assert records[0]["annotations"][0]["box_id"] == "sample-1:box:0"
    assert records[0]["source"]["queue_path"] == str(manifest)
    assert records[0]["decision_id"]
    assert records[1]["previous_status"] == "UNCERTAIN"
    assert records[1]["new_status"] == "REJECTED"
    assert duplicate["saved"] is False
    assert duplicate["duplicate"] is True


def test_reviewer_type_must_be_explicit_human():
    base = {
        "sample_id": "sample-1",
        "reviewer_id": "reviewer@example.org",
        "status": "UNCERTAIN",
        "decision_reason": "INSUFFICIENT_EVIDENCE",
        "annotations": [],
    }
    with pytest.raises(ValueError):
        reviewer.Decision(**base)
    with pytest.raises(ValueError):
        reviewer.Decision(**base, reviewer_type="AI")
