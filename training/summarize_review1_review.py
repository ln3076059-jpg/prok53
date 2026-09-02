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


def _summarize_lane(queue: list[dict], decisions: list[dict]) -> dict:
    review1_by_id = {}
    for item in decisions:
        sample_id = str(item["sample_id"])
        if item.get("reviewer_id") == "review1" and item.get("reviewer_type") == "AI":
            review1_by_id[sample_id] = item
    reviewed = list(review1_by_id.values())
    tiers = Counter(item.get("tier", "TIER_C") for item in reviewed)
    statuses = Counter(item.get("new_status") or item.get("status") for item in reviewed)
    
    return {
        "total": len(queue),
        "reviewed": len(reviewed),
        "tier_a": tiers["TIER_A"],
        "tier_b": tiers["TIER_B"],
        "tier_c": tiers["TIER_C"],
        "accepted": statuses["REVIEW1_APPROVED_UNDER_ADMIN_DELEGATION"] + statuses["REVIEW1_ACCEPTED_PROPOSAL"],
        "rejected": statuses["REVIEW1_REJECTED_PROPOSAL"],
    }


def summarize(
    neg_queue: list[dict],
    neg_decisions: list[dict],
    pos_queue: list[dict],
    pos_decisions: list[dict],
    attention: list[dict],
    head_sha: str,
) -> dict:
    remaining = len({str(item["sample_id"]) for item in attention})
    return {
        "audited_source_head_sha": head_sha,
        "phone_negative": _summarize_lane(neg_queue, neg_decisions),
        "phone_positive": _summarize_lane(pos_queue, pos_decisions),
        "global_attention": {
            "remaining": remaining
        },
        "governed_ready": False,
    }


def markdown(report: dict) -> str:
    lines = [
        "# Review 1 full review summary",
        "",
        "Review 1 records are AI provenance under admin delegation. They are not human approval.",
        "",
        f"- audited_source_head_sha: `{report['audited_source_head_sha']}`",
        f"- governed_ready: `{report['governed_ready']}`",
        "",
        "## Phone Negative",
    ]
    for k, v in report["phone_negative"].items():
        lines.append(f"- {k}: `{v}`")
    lines.extend([
        "",
        "## Phone Positive"
    ])
    for k, v in report["phone_positive"].items():
        lines.append(f"- {k}: `{v}`")
    lines.extend([
        "",
        "## Global Attention",
        f"- remaining: `{report['global_attention']['remaining']}`",
        ""
    ])
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize append-only Review 1 evidence")
    parser.add_argument("--queue", type=Path, help="legacy arg ignored")
    parser.add_argument("--decisions", type=Path, help="legacy arg ignored")
    parser.add_argument("--attention", type=Path, default=Path("datasets/manifests/v2_human_attention_after_review1.json"))
    parser.add_argument("--json-output", type=Path, default=Path("reports/REVIEW1_FULL_REVIEW_SUMMARY.json"))
    parser.add_argument("--markdown-output", type=Path, default=Path("reports/REVIEW1_FULL_REVIEW_SUMMARY.md"))
    args = parser.parse_args()

    def load_q(p: Path) -> list[dict]:
        if not p.exists(): return []
        payload = json.loads(p.read_text(encoding="utf-8"))
        return payload if isinstance(payload, list) else payload.get("samples", [])

    neg_queue = load_q(Path("datasets/manifests/v2_phone_negative_review.json"))
    neg_decisions = read_jsonl(Path("datasets/manifests/review1_phone_negative_decisions.jsonl"))
    
    pos_queue = load_q(Path("datasets/manifests/v2_phone_positive_review.json"))
    pos_decisions = read_jsonl(Path("datasets/manifests/review1_phone_positive_decisions.jsonl"))

    attention = load_q(args.attention)

    head = subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()
    report = summarize(
        neg_queue,
        neg_decisions,
        pos_queue,
        pos_decisions,
        attention,
        head,
    )
    args.json_output.parent.mkdir(parents=True, exist_ok=True)
    args.json_output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(markdown(report), encoding="utf-8")
    print(json.dumps(report, indent=2))

if __name__ == "__main__":
    main()
