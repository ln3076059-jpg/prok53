from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from training.common import sha256_file

ADVERSE_CONDITIONS = {"low_light", "motion_blur", "reflection", "partial_occlusion"}
PHONE_LIKE_CLASSES = {
    "cigarette",
    "drinking",
    "distraction",
    "screen",
    "wallet",
}


def _priority(sample: dict) -> tuple[int, list[str]]:
    review = sample.get("proposal_review", {})
    checks = review.get("checks", {})
    phone_model = checks.get("phone_model", {})
    conditions = set(checks.get("image_quality", {}).get("suggested_conditions", []))
    source_classes = {
        str(item.get("source_class_name", "")).strip().lower()
        for item in sample.get("excluded_source_annotations", [])
    }
    reasons: list[str] = []
    priority = 4
    if int(phone_model.get("detected_phones", 0)) > 0:
        priority = 1
        reasons.append("MODEL_PREDICTS_PHONE_WITHOUT_SOURCE_PHONE_BOX")
    if conditions & ADVERSE_CONDITIONS:
        priority = min(priority, 2)
        reasons.append("ADVERSE_VISIBILITY_REQUIRES_HUMAN_CHECK")
    if source_classes & PHONE_LIKE_CLASSES:
        priority = min(priority, 2)
        reasons.append("PHONE_LIKE_OBJECT_OR_ACTION")
    if review.get("decision") == "UNCERTAIN" or review.get("tier") == "UNCERTAIN":
        priority = min(priority, 2)
        reasons.append("MODEL_ASSISTED_REVIEW_UNCERTAIN")
    if not reasons:
        reasons.append("ZERO_BOX_NEGATIVE_REQUIRES_HUMAN_CONFIRMATION")
    return priority, reasons


def build(source_report: Path, output: Path, summary_output: Path) -> dict:
    payload = json.loads(source_report.read_text(encoding="utf-8"))
    queue_path = Path(payload["queue"])
    if not queue_path.is_file() or sha256_file(queue_path) != payload["queue_sha256"]:
        raise ValueError("model-assisted report is stale for its source review queue")
    queue_payload = json.loads(queue_path.read_text(encoding="utf-8"))
    queue_items = (
        queue_payload if isinstance(queue_payload, list) else queue_payload.get("selected", [])
    )
    if len(queue_items) != int(payload.get("queue_samples", -1)):
        raise ValueError("source review queue sample count does not match the report")
    samples = payload.get("samples", [])
    if len(samples) != int(payload.get("queue_samples", -1)):
        raise ValueError("model-assisted report does not cover the complete phone-negative queue")
    if {item["sample_id"] for item in samples} != {item["sample_id"] for item in queue_items}:
        raise ValueError("model-assisted report sample identities do not match the source queue")
    prioritized = []
    for sample in samples:
        priority, reasons = _priority(sample)
        prioritized.append(
            {
                **sample,
                "review_status": "PENDING",
                "human_review_status": "PENDING",
                "automatic_training_label": None,
                "review_priority": priority,
                "review_priority_reasons": reasons,
                "review_tasks": sorted(
                    {
                        *sample.get("review_tasks", []),
                        "MISSING_PHONE_LABEL_REVIEW",
                        "PHONE_LIKE_FALSE_POSITIVE_REVIEW",
                        "MOUNTED_OR_PASSENGER_CONTEXT_REVIEW",
                        "CROP_QUALITY_REVIEW",
                    }
                ),
            }
        )
    prioritized.sort(key=lambda item: (item["review_priority"], str(item["sample_id"])))
    if any(item["review_status"] != "PENDING" for item in prioritized):
        raise AssertionError("phone-negative priority queue must remain PENDING")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(prioritized, indent=2, sort_keys=True), encoding="utf-8")
    counts = Counter(str(item["review_priority"]) for item in prioritized)
    reason_counts = Counter(
        reason for item in prioritized for reason in item["review_priority_reasons"]
    )
    summary = {
        "schema_version": 1,
        "status": "HUMAN_REVIEW_REQUIRED",
        "samples": len(prioritized),
        "priority_counts": dict(sorted(counts.items())),
        "priority_reason_counts": dict(sorted(reason_counts.items())),
        "source_report": str(source_report),
        "source_report_sha256": sha256_file(source_report),
        "source_queue": str(queue_path),
        "source_queue_sha256": sha256_file(queue_path),
        "output_queue": str(output),
        "output_queue_sha256": sha256_file(output),
        "human_approved": 0,
        "policy": "Priority is model-assisted triage only; every final decision remains human.",
    }
    summary_output.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Prioritize zero-box phone proposals for humans")
    parser.add_argument(
        "--source-report",
        type=Path,
        default=Path("datasets/manifests/pretrain_pending_approval/v2_phone_negative.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/manifests/v2_phone_negative_priority_review.json"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/v2_phone_negative_priority_review.json"),
    )
    args = parser.parse_args()
    print(json.dumps(build(args.source_report, args.output, args.summary), indent=2))


if __name__ == "__main__":
    main()
