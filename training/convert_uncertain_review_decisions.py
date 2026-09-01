from __future__ import annotations

import argparse
import json
from pathlib import Path

CLASS_DECISIONS = {1: "SEATBELT_FASTENED", 2: "SEATBELT_UNFASTENED"}


def _jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def convert(queue_path: Path, decisions_path: Path, output: Path, reviewer: str) -> dict:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    by_id = {item["sample_id"]: item for item in queue}
    latest = {}
    for decision in _jsonl(decisions_path):
        sample_id = decision["sample_id"]
        if sample_id not in by_id:
            raise ValueError(f"unknown uncertainty review sample: {sample_id}")
        latest[sample_id] = decision
    converted = []
    for sample_id, decision in sorted(latest.items()):
        status = decision["status"]
        if status == "APPROVED":
            annotations = decision.get("annotations", [])
            if len(annotations) != 1:
                raise ValueError(
                    f"approved uncertainty candidate needs exactly one upper-body box: {sample_id}"
                )
            class_id = int(annotations[0]["class_id"])
            if class_id not in CLASS_DECISIONS:
                raise ValueError(
                    f"uncertainty candidate cannot become class {class_id}: {sample_id}"
                )
            value = CLASS_DECISIONS[class_id]
        elif status == "UNCERTAIN":
            value = "UNCERTAIN_OR_OCCLUDED"
        elif status in {"REJECTED", "APPROVED_NEGATIVE"}:
            value = "REJECT_BAD_ROI"
        else:
            raise ValueError(f"unsupported reviewer status {status}: {sample_id}")
        converted.append(
            {
                "candidate_id": by_id[sample_id]["candidate_id"],
                "decision": value,
                "notes": decision.get("notes", ""),
                "reviewed_at": decision["reviewed_at_utc"],
                "reviewer": reviewer,
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in converted)
        + ("\n" if converted else ""),
        encoding="utf-8",
    )
    return {
        "queue_samples": len(queue),
        "converted_decisions": len(converted),
        "pending": len(queue) - len(converted),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert reviewer UI decisions for belt uncertainty"
    )
    parser.add_argument("decisions", type=Path)
    parser.add_argument("--reviewer", required=True)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("datasets/manifests/v2_seatbelt_uncertain_ui_review.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/manifests/seatbelt_uncertain_decisions_v2.jsonl"),
    )
    args = parser.parse_args()
    print(json.dumps(convert(args.queue, args.decisions, args.output, args.reviewer), indent=2))


if __name__ == "__main__":
    main()
