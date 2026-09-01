from __future__ import annotations

import json

from training.common import sha256_file
from training.prioritize_phone_negative_review import build


def test_phone_negative_priority_keeps_model_findings_pending(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    samples = [
        {
            "sample_id": "ordinary",
            "review_tasks": [],
            "proposal_review": {
                "decision": "PASS",
                "tier": "MEDIUM_CONFIDENCE",
                "checks": {
                    "phone_model": {"detected_phones": 0},
                    "image_quality": {"suggested_conditions": []},
                },
            },
        },
        {
            "sample_id": "missing-phone",
            "review_tasks": [],
            "proposal_review": {
                "decision": "UNCERTAIN",
                "tier": "UNCERTAIN",
                "checks": {
                    "phone_model": {"detected_phones": 1},
                    "image_quality": {"suggested_conditions": ["low_light"]},
                },
            },
        },
    ]
    queue = tmp_path / "queue.json"
    queue.write_text(
        json.dumps([{"sample_id": item["sample_id"]} for item in samples]), encoding="utf-8"
    )
    source = tmp_path / "source.json"
    source.write_text(
        json.dumps(
            {
                "queue": str(queue),
                "queue_sha256": sha256_file(queue),
                "queue_samples": 2,
                "samples": samples,
            }
        ),
        encoding="utf-8",
    )
    output = tmp_path / "priority.json"
    summary = build(source, output, tmp_path / "summary.json")
    prioritized = json.loads(output.read_text(encoding="utf-8"))

    assert prioritized[0]["sample_id"] == "missing-phone"
    assert prioritized[0]["review_status"] == "PENDING"
    assert prioritized[0]["automatic_training_label"] is None
    assert summary["human_approved"] == 0
