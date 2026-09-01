from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import yaml

from training.common import sha256_file


def _load_items(path: Path) -> list[dict]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".jsonl":
        return [json.loads(line) for line in text.splitlines() if line.strip()]
    payload = json.loads(text)
    if isinstance(payload, list):
        return payload
    for key in ("selected", "items", "samples"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError(
        "capture manifest must be a JSON list, JSONL, or contain selected/items/samples"
    )


def build(
    manifest_path: Path,
    policy_path: Path,
    output_path: Path,
    per_group: int = 1,
    require_complete_coverage: bool = False,
) -> dict:
    if per_group < 1:
        raise ValueError("per_group must be positive")
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    required_fields = set(policy["required_fields"])
    required_scenarios = set(policy["required_scenarios"])
    allowed_statuses = set(policy.get("allowed_review_statuses", ["PENDING"]))
    items = _load_items(manifest_path)
    errors: list[str] = []
    seen_sha: dict[str, str] = {}
    candidates: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)

    for index, item in enumerate(items, start=1):
        label = str(item.get("sample_id") or f"row {index}")
        missing = sorted(field for field in required_fields if item.get(field) in (None, ""))
        if missing:
            errors.append(f"{label}: missing required fields: {', '.join(missing)}")
            continue
        scenario = str(item["scenario"])
        if scenario not in required_scenarios:
            errors.append(f"{label}: unsupported hard-negative scenario: {scenario}")
        if item["review_status"] not in allowed_statuses:
            errors.append(f"{label}: review_status must remain PENDING before human review")
        if item.get("automatic_training_label") is not None:
            errors.append(f"{label}: automatic_training_label must be null")
        if not isinstance(item["conditions"], list) or not item["conditions"]:
            errors.append(f"{label}: conditions must be a non-empty list")
        try:
            captured_at = datetime.fromisoformat(str(item["captured_at"]).replace("Z", "+00:00"))
            if captured_at.tzinfo is None:
                raise ValueError
        except ValueError:
            errors.append(f"{label}: captured_at must be a timezone-aware ISO-8601 timestamp")
        image_path = Path(item["image_path"])
        if not image_path.is_file():
            errors.append(f"{label}: image file does not exist: {image_path}")
            continue
        actual_sha = sha256_file(image_path)
        if actual_sha != item["sha256"]:
            errors.append(f"{label}: SHA-256 does not match image_path")
            continue
        if previous := seen_sha.get(actual_sha):
            errors.append(f"{label}: byte-identical image already declared by {previous}")
            continue
        seen_sha[actual_sha] = label
        candidates[(scenario, str(item["source_group_id"]))].append(item)

    if errors:
        raise ValueError(
            "invalid seatbelt hard-negative capture manifest:\n- " + "\n- ".join(errors)
        )

    selected: list[dict] = []
    group_counts: Counter[str] = Counter()
    for (scenario, group_id), group_items in sorted(candidates.items()):
        for item in sorted(group_items, key=lambda value: str(value["sample_id"]))[:per_group]:
            selected.append(
                {
                    **item,
                    "source_group_id": group_id,
                    "scenario": scenario,
                    "review_status": "PENDING",
                    "human_review_status": "PENDING",
                    "annotations": [],
                    "automatic_training_label": None,
                    "allowed_decisions": [
                        "CONFIRMED_HARD_NEGATIVE",
                        "OCCUPANT_PRESENT_REANNOTATE",
                        "UNCERTAIN",
                        "REJECT",
                    ],
                    "review_tasks": [
                        "VEHICLE_CONTEXT_REVIEW",
                        "OCCUPANT_PRESENCE_REVIEW",
                        "SEATBELT_ROI_HARD_NEGATIVE_DECISION",
                    ],
                }
            )
            group_counts[scenario] += 1

    missing_scenarios = sorted(required_scenarios - set(group_counts))
    if require_complete_coverage and missing_scenarios:
        raise ValueError(
            "hard-negative capture coverage is incomplete: " + ", ".join(missing_scenarios)
        )
    report = {
        "schema_version": 1,
        "status": (
            "HUMAN_REVIEW_REQUIRED" if not missing_scenarios else "CAPTURE_COVERAGE_INCOMPLETE"
        ),
        "selection_policy": "AT_MOST_N_SAMPLES_PER_SOURCE_GROUP_AND_SCENARIO",
        "per_group": per_group,
        "selected_samples": len(selected),
        "independent_group_counts": dict(sorted(group_counts.items())),
        "missing_required_scenarios": missing_scenarios,
        "selected": selected,
        "policy": (
            "All items are unlabelled proposals. Only append-only human decisions may make "
            "confirmed hard negatives training-eligible."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a governed seatbelt ROI hard-negative human-review queue"
    )
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("datasets/v2_seatbelt_hard_negative_policy.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/manifests/v2_seatbelt_hard_negative_review.json"),
    )
    parser.add_argument("--per-group", type=int, default=1)
    parser.add_argument("--require-complete-coverage", action="store_true")
    args = parser.parse_args()
    report = build(
        args.manifest,
        args.policy,
        args.output,
        args.per_group,
        args.require_complete_coverage,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "selected"}, indent=2))


if __name__ == "__main__":
    main()
