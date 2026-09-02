from __future__ import annotations

import argparse
import json
import subprocess
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _changed_annotations(record: dict) -> tuple[bool, bool]:
    original = record.get("original_annotations", [])
    reviewed = record.get("reviewed_annotations", record.get("annotations", []))
    bbox_changed = len(original) != len(reviewed) or any(
        left.get("yolo") != right.get("yolo")
        for left, right in zip(original, reviewed, strict=False)
    )
    class_changed = len(original) == len(reviewed) and any(
        left.get("class_id") != right.get("class_id")
        for left, right in zip(original, reviewed, strict=True)
    )
    return bbox_changed, class_changed


def summarize(
    queue: list[dict],
    decisions: list[dict],
    acknowledgements: list[dict],
    attention: list[dict],
    head_sha: str,
) -> dict:
    review1_by_id = {}
    human_confirmed = set()
    for item in decisions:
        sample_id = str(item["sample_id"])
        if item.get("reviewer_id") == "review1" and item.get("reviewer_type") == "AI":
            review1_by_id[sample_id] = item
        if item.get("reviewer_type") == "HUMAN" and item.get("status") in {
            "APPROVED",
            "APPROVED_NEGATIVE",
        }:
            human_confirmed.add(sample_id)
    reviewed = list(review1_by_id.values())
    tiers = Counter(item.get("tier", "TIER_C") for item in reviewed)
    statuses = Counter(item.get("new_status") or item.get("status") for item in reviewed)
    bbox_corrections = 0
    class_corrections = 0
    for item in reviewed:
        bbox_changed, class_changed = _changed_annotations(item)
        bbox_corrections += bbox_changed
        class_corrections += class_changed
    acknowledgement_statuses = {
        "ADMIN_ACKNOWLEDGED_MODEL_PROPOSAL_BATCH",
        "ADMIN_ACKNOWLEDGED_REVIEW_PENDING_APPROVAL_BATCH",
    }
    acknowledged = {
        str(sample_id)
        for item in acknowledgements
        if item.get("status") in acknowledgement_statuses
        for sample_id in item.get("sample_ids", [])
    }
    acknowledged_records = sum(
        len(item.get("sample_ids", []))
        for item in acknowledgements
        if item.get("status") in acknowledgement_statuses
    )
    total_queue = len(queue)
    remaining = len({str(item["sample_id"]) for item in attention})
    reduction = 0.0 if not total_queue else 100 * max(0, total_queue - remaining) / total_queue
    return {
        "head_sha": head_sha,
        "total_queue": total_queue,
        "review1_reviewed": len(reviewed),
        "tier_a": tiers["TIER_A"],
        "tier_b": tiers["TIER_B"],
        "tier_c": tiers["TIER_C"],
        "review1_accepted": statuses["REVIEW1_APPROVED_UNDER_ADMIN_DELEGATION"]
        + statuses["REVIEW1_ACCEPTED_PROPOSAL"],
        "review1_rejected": statuses["REVIEW1_REJECTED_PROPOSAL"],
        "review1_corrections": statuses["REVIEW1_CORRECTION_PROPOSAL"],
        "missing_phone_labels": sum(
            item.get("decision_reason") == "PHONE_MISSING_LABEL" for item in reviewed
        ),
        "bbox_corrections": bbox_corrections,
        "class_corrections": class_corrections,
        "uncertain_seatbelt": sum(
            item.get("decision_reason") == "UNCERTAIN_OR_OCCLUDED" for item in reviewed
        ),
        "unknown_occupant": sum(
            item.get("occupant_role") in {None, "UNKNOWN", "UNCERTAIN", "PENDING"}
            for item in reviewed
        ),
        "bad_metadata": sum(
            any(item.get(key) in {None, "", "UNKNOWN", "NOT_PROVABLE"} for key in (
                "video_id",
                "vehicle_id",
                "person_id",
                "camera_id",
            ))
            for item in reviewed
        ),
        "admin_acknowledged": acknowledged_records,
        "admin_acknowledged_unique_samples": len(acknowledged),
        "human_confirmed": len(human_confirmed),
        "remaining_human_attention": remaining,
        "manual_review_reduction_percent": round(reduction, 2),
        "governed_ready": False,
    }


def markdown(report: dict) -> str:
    labels = {
        "head_sha": "HEAD SHA",
        "total_queue": "TOTAL_QUEUE",
        "review1_reviewed": "REVIEW1_REVIEWED",
        "tier_a": "TIER_A",
        "tier_b": "TIER_B",
        "tier_c": "TIER_C",
        "review1_accepted": "REVIEW1_ACCEPTED",
        "review1_rejected": "REVIEW1_REJECTED",
        "review1_corrections": "REVIEW1_CORRECTIONS",
        "missing_phone_labels": "MISSING_PHONE_LABELS",
        "bbox_corrections": "BBOX_CORRECTIONS",
        "class_corrections": "CLASS_CORRECTIONS",
        "uncertain_seatbelt": "UNCERTAIN_SEATBELT",
        "unknown_occupant": "UNKNOWN_OCCUPANT",
        "bad_metadata": "BAD_METADATA",
        "admin_acknowledged": "ADMIN_ACKNOWLEDGED",
        "admin_acknowledged_unique_samples": "ADMIN_ACKNOWLEDGED_UNIQUE_SAMPLES",
        "human_confirmed": "HUMAN_CONFIRMED",
        "remaining_human_attention": "REMAINING_HUMAN_ATTENTION",
        "manual_review_reduction_percent": "MANUAL_REVIEW_REDUCTION_PERCENT",
        "governed_ready": "GOVERNED_READY",
    }
    lines = [
        "# Review 1 full review summary",
        "",
        "Review 1 records are AI provenance under admin delegation. They are not human approval.",
        "",
    ]
    lines.extend(f"- {labels[key]}: `{value}`" for key, value in report.items())
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize append-only Review 1 evidence")
    parser.add_argument("--queue", type=Path, required=True)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("datasets/manifests/review_decisions.jsonl"),
    )
    parser.add_argument(
        "--acknowledgements",
        type=Path,
        default=Path("datasets/manifests/model_proposal_batch_acknowledgements.jsonl"),
    )
    parser.add_argument(
        "--attention",
        type=Path,
        default=Path("datasets/manifests/v2_human_attention_after_review1.json"),
    )
    parser.add_argument(
        "--json-output", type=Path, default=Path("reports/REVIEW1_FULL_REVIEW_SUMMARY.json")
    )
    parser.add_argument(
        "--markdown-output", type=Path, default=Path("reports/REVIEW1_FULL_REVIEW_SUMMARY.md")
    )
    args = parser.parse_args()
    queue_payload = json.loads(args.queue.read_text(encoding="utf-8"))
    queue = queue_payload if isinstance(queue_payload, list) else queue_payload.get("samples", [])
    attention = (
        json.loads(args.attention.read_text(encoding="utf-8"))
        if args.attention.exists()
        else []
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = summarize(
        queue,
        read_jsonl(args.decisions),
        read_jsonl(args.acknowledgements),
        attention,
        head,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
