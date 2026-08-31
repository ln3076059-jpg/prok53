from __future__ import annotations

import argparse
import json
from copy import deepcopy
from pathlib import Path

from training.common import YoloLabel


def _annotation(proposal: dict, model: str) -> dict:
    if proposal.get("class_id") != 0 or proposal.get("class_name") != "phone":
        raise ValueError("proposal must be canonical class 0 / phone")
    label = YoloLabel(0, *(float(value) for value in proposal["yolo"]))
    label.validate()
    confidence = float(proposal["confidence"])
    if not 0 <= confidence <= 1:
        raise ValueError("proposal confidence is outside [0, 1]")
    return {
        "class_id": 0,
        "yolo": [round(value, 8) for value in proposal["yolo"]],
        "confidence": confidence,
        "proposal_source": model,
        "occupant_role": "PENDING",
        "requires_review": True,
        "review_status": "PENDING",
    }


def merge(queue: list[dict], report: dict, keep_all_queue: bool = False) -> list[dict]:
    generator = report.get("generator", {})
    model = str(generator.get("model", "")).strip()
    if not model:
        raise ValueError("proposal report is missing generator.model")
    proposal_samples = report.get("samples")
    if not isinstance(proposal_samples, list):
        raise ValueError("proposal report is missing samples")
    by_id = {item["sample_id"]: item for item in queue}
    if len(by_id) != len(queue):
        raise ValueError("review queue contains duplicate sample_id values")
    proposal_ids = [item.get("sample_id") for item in proposal_samples]
    if len(set(proposal_ids)) != len(proposal_ids):
        raise ValueError("proposal report contains duplicate sample_id values")
    unknown = sorted(set(proposal_ids) - by_id.keys())
    if unknown:
        raise ValueError(f"proposal samples are absent from review queue: {unknown[:5]}")

    selected = queue if keep_all_queue else [by_id[sample_id] for sample_id in proposal_ids]
    proposals_by_id = {item["sample_id"]: item for item in proposal_samples}
    merged: list[dict] = []
    for original in selected:
        item = deepcopy(original)
        sample = proposals_by_id.get(item["sample_id"])
        if sample is not None:
            if sample.get("source_asset_id") != item.get("source_asset_id"):
                raise ValueError(f"source asset mismatch for {item['sample_id']}")
            proposed = [_annotation(value, model) for value in sample.get("proposals", [])]
            existing = item.get("annotations", [])
            if existing:
                raise ValueError(f"refusing to replace existing annotations for {item['sample_id']}")
            item["annotations"] = proposed
            item["proposal_count"] = len(proposed)
            item["proposal_generator"] = deepcopy(generator)
            item["occupant_role"] = "PENDING"
            tasks = list(dict.fromkeys([*item.get("review_tasks", []), "PHYSICAL_PHONE_REBOX", "OCCUPANT_ROLE_REVIEW"]))
            item["review_tasks"] = tasks
            item["notes"] = (
                "Bootstrap boxes are proposals only. Rebox/delete as needed; confirm vehicle context and "
                "driver/passenger role. " + item.get("notes", "")
            ).strip()
        merged.append(item)
    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge phone proposals into a human-review queue")
    parser.add_argument("review_queue", type=Path)
    parser.add_argument("proposal_report", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--keep-all-queue", action="store_true")
    args = parser.parse_args()
    queue = json.loads(args.review_queue.read_text(encoding="utf-8"))
    report = json.loads(args.proposal_report.read_text(encoding="utf-8"))
    result = merge(queue, report, args.keep_all_queue)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"output": str(args.output), "review_samples": len(result), "proposed_boxes": sum(len(item.get("annotations", [])) for item in result)}, indent=2))


if __name__ == "__main__":
    main()
