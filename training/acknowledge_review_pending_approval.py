from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha256_file

STATUS = "ADMIN_ACKNOWLEDGED_REVIEW_PENDING_APPROVAL_BATCH"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def acknowledge(
    review_path: Path,
    summary_path: Path,
    output: Path,
    reviewer_id: str,
    batch_size: int,
) -> dict:
    reviewer_id = reviewer_id.strip()
    if len(reviewer_id) < 2:
        raise ValueError("reviewer_id must contain at least two characters")
    if not 1 <= batch_size <= 500:
        raise ValueError("batch_size must be between 1 and 500")
    rows = _read_jsonl(review_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    if summary.get("status") != "INTEGRITY_PASS":
        raise ValueError("pretrain review summary integrity status is not INTEGRITY_PASS")
    if summary.get("images_checked") != len(rows):
        raise ValueError("pretrain review row count does not match its summary")
    if summary.get("reviewer_type") != "AUTOMATED":
        raise ValueError("source review must retain automated-review provenance")
    if summary.get("approval_status") != "PENDING_HUMAN_APPROVAL":
        raise ValueError("source review must remain pending human approval")

    review_hash = sha256_file(review_path)
    summary_hash = sha256_file(summary_path)
    existing = _read_jsonl(output) if output.is_file() else []
    matching = [
        item
        for item in existing
        if item.get("source_review_sha256") == review_hash
        and item.get("reviewer_id") == reviewer_id
        and item.get("status") == STATUS
    ]
    if matching:
        acknowledged_keys = {
            key for item in matching for key in item.get("worklist_keys", [])
        }
        if acknowledged_keys != {item["worklist_key"] for item in rows}:
            raise ValueError("existing admin acknowledgement is incomplete or inconsistent")
        return {
            "status": "REUSED_EXISTING_ACKNOWLEDGEMENT",
            "reviewer_id": reviewer_id,
            "acknowledged_samples": len(acknowledged_keys),
            "batches": len(matching),
            "output": str(output),
            "governance_eligible": False,
        }

    output.parent.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(UTC).isoformat()
    records = []
    for batch_index, start in enumerate(range(0, len(rows), batch_size), 1):
        batch = rows[start : start + batch_size]
        records.append(
            {
                "schema_version": 1,
                "status": STATUS,
                "reviewer_type": "HUMAN",
                "reviewer_id": reviewer_id,
                "acknowledged_at_utc": timestamp,
                "batch_index": batch_index,
                "sample_ids": [item["sample_id"] for item in batch],
                "worklist_keys": [item["worklist_key"] for item in batch],
                "source_review": str(review_path),
                "source_review_sha256": review_hash,
                "source_summary": str(summary_path),
                "source_summary_sha256": summary_hash,
                "acknowledgement_scope": "ADMIN_ACKNOWLEDGED_PENDING_REVIEW_AS_A_BATCH",
                "per_sample_human_inspection_claimed": False,
                "human_approved": False,
                "governance_eligible": False,
                "policy": (
                    "A human admin acknowledged receipt of the pending pretrain review as a "
                    "batch. This is not evidence that the admin visually inspected every "
                    "image and does not create per-sample HUMAN_APPROVED ground truth."
                ),
            }
        )
    with output.open("a", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, sort_keys=True) + "\n")
    return {
        "status": "ACKNOWLEDGED",
        "reviewer_id": reviewer_id,
        "acknowledged_samples": len(rows),
        "batches": len(records),
        "output": str(output),
        "source_review_sha256": review_hash,
        "governance_eligible": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Record admin acknowledgement of a review pending human approval"
    )
    parser.add_argument(
        "--review",
        type=Path,
        default=Path("reports/v2_review_pending_approval.jsonl"),
    )
    parser.add_argument(
        "--summary",
        type=Path,
        default=Path("reports/v2_review_pending_approval.summary.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "datasets/manifests/review_pending_approval_admin_acknowledgements.jsonl"
        ),
    )
    parser.add_argument("--reviewer-id", default="admin")
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()
    print(
        json.dumps(
            acknowledge(
                args.review,
                args.summary,
                args.output,
                args.reviewer_id,
                args.batch_size,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
