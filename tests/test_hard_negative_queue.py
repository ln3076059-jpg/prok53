from __future__ import annotations

import json

import pytest

from training.build_seatbelt_hard_negative_queue import build
from training.common import sha256_file


def _policy(path, scenarios: list[str]) -> None:
    scenario_list = ", ".join(scenarios)
    path.write_text(
        f"""schema_version: 1
required_fields:
  - sample_id
  - image_path
  - sha256
  - source_id
  - source_asset_id
  - source_group_id
  - camera_id
  - scenario
  - conditions
  - captured_at
  - review_status
required_scenarios: [{scenario_list}]
allowed_review_statuses: [PENDING]
""",
        encoding="utf-8",
    )


def _item(image_path, sample_id: str = "sample-1", scenario: str = "empty_seat") -> dict:
    return {
        "sample_id": sample_id,
        "image_path": str(image_path),
        "sha256": sha256_file(image_path),
        "source_id": "independent-capture",
        "source_asset_id": f"asset-{sample_id}",
        "source_group_id": f"group-{sample_id}",
        "camera_id": "camera-1",
        "scenario": scenario,
        "conditions": ["daylight"],
        "captured_at": "2026-09-02T00:00:00Z",
        "review_status": "PENDING",
    }


def test_hard_negative_queue_preserves_pending_unlabelled_proposals(tmp_path):
    image = tmp_path / "empty-seat.jpg"
    image.write_bytes(b"independent-empty-seat-capture")
    manifest = tmp_path / "capture.json"
    policy = tmp_path / "policy.yaml"
    output = tmp_path / "queue.json"
    manifest.write_text(json.dumps([_item(image)]), encoding="utf-8")
    _policy(policy, ["empty_seat"])

    report = build(manifest, policy, output, require_complete_coverage=True)

    assert report["status"] == "HUMAN_REVIEW_REQUIRED"
    assert report["selected_samples"] == 1
    candidate = report["selected"][0]
    assert candidate["review_status"] == "PENDING"
    assert candidate["human_review_status"] == "PENDING"
    assert candidate["automatic_training_label"] is None
    assert candidate["annotations"] == []


def test_hard_negative_queue_rejects_auto_approval_and_hash_mismatch(tmp_path):
    image = tmp_path / "reflection.jpg"
    image.write_bytes(b"reflection-capture")
    item = _item(image, scenario="reflection")
    item["review_status"] = "APPROVED"
    item["sha256"] = "f" * 64
    item["captured_at"] = "not-a-timestamp"
    manifest = tmp_path / "capture.json"
    policy = tmp_path / "policy.yaml"
    manifest.write_text(json.dumps([item]), encoding="utf-8")
    _policy(policy, ["reflection"])

    with pytest.raises(ValueError, match="review_status must remain PENDING") as exc:
        build(manifest, policy, tmp_path / "queue.json")
    assert "SHA-256 does not match" in str(exc.value)
    assert "timezone-aware ISO-8601" in str(exc.value)


def test_hard_negative_queue_reports_missing_capture_coverage(tmp_path):
    image = tmp_path / "empty-seat.jpg"
    image.write_bytes(b"one-scenario-only")
    manifest = tmp_path / "capture.jsonl"
    policy = tmp_path / "policy.yaml"
    manifest.write_text(json.dumps(_item(image)) + "\n", encoding="utf-8")
    _policy(policy, ["empty_seat", "windshield_glare"])

    report = build(manifest, policy, tmp_path / "queue.json")
    assert report["status"] == "CAPTURE_COVERAGE_INCOMPLETE"
    assert report["missing_required_scenarios"] == ["windshield_glare"]

    with pytest.raises(ValueError, match="capture coverage is incomplete"):
        build(
            manifest,
            policy,
            tmp_path / "strict-queue.json",
            require_complete_coverage=True,
        )
