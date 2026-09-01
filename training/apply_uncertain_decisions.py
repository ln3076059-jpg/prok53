from __future__ import annotations

import argparse
import json
from pathlib import Path


def _jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def apply(queue_path: Path, decisions_path: Path, output: Path) -> dict:
    queue = json.loads(queue_path.read_text(encoding="utf-8"))["selected"]
    by_id = {item["candidate_id"]: item for item in queue}
    decisions = _jsonl(decisions_path)
    seen = set()
    accepted = []
    allowed = {
        "SEATBELT_FASTENED",
        "SEATBELT_UNFASTENED",
        "UNCERTAIN_OR_OCCLUDED",
        "REJECT_BAD_ROI",
    }
    for decision in decisions:
        candidate_id = decision.get("candidate_id")
        if candidate_id not in by_id:
            raise ValueError(f"unknown candidate_id: {candidate_id}")
        if candidate_id in seen:
            raise ValueError(f"duplicate decision: {candidate_id}")
        seen.add(candidate_id)
        value = decision.get("decision")
        if value not in allowed:
            raise ValueError(f"invalid decision for {candidate_id}: {value}")
        if not decision.get("reviewer") or not decision.get("reviewed_at"):
            raise ValueError(f"decision lacks reviewer/audit timestamp: {candidate_id}")
        if value != "UNCERTAIN_OR_OCCLUDED":
            continue
        item = by_id[candidate_id]
        accepted.append(
            {
                "sample_id": item["candidate_id"].replace(":", "_"),
                "source_sample_id": item["sample_id"],
                "source_box_index": item["box_index"],
                "split": item["split"],
                "image_path": item["image_path"],
                "sha256": item["sha256"],
                "yolo": item["yolo"],
                "source_group_id": item["source_group_id"],
                "effective_group_id": item["effective_group_id"],
                "seatbelt_state": "uncertain_or_occluded",
                "reviewer": decision["reviewer"],
                "reviewed_at": decision["reviewed_at"],
                "review_notes": decision.get("notes"),
            }
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in accepted)
        + ("\n" if accepted else ""),
        encoding="utf-8",
    )
    return {
        "queue_candidates": len(queue),
        "decisions": len(decisions),
        "uncertain_accepted": len(accepted),
        "pending": len(queue) - len(seen),
        "output": str(output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize reviewed V2 uncertain crops")
    parser.add_argument("decisions", type=Path)
    parser.add_argument(
        "--queue",
        type=Path,
        default=Path("datasets/manifests/seatbelt_uncertain_review_v2.json"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/manifests/seatbelt_uncertain_v2.jsonl")
    )
    args = parser.parse_args()
    print(json.dumps(apply(args.queue, args.decisions, args.output), indent=2))


if __name__ == "__main__":
    main()
