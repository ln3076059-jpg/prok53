from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

import yaml

from training.common import sha256_file


def _valid_yolo(values: list[float]) -> bool:
    if len(values) != 4:
        return False
    x, y, width, height = (float(value) for value in values)
    return (
        width > 0
        and height > 0
        and x - width / 2 >= 0
        and x + width / 2 <= 1
        and y - height / 2 >= 0
        and y + height / 2 <= 1
    )


def validate(manifest_path: Path, policy_path: Path) -> dict:
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    records = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line
    ]
    errors: list[str] = []
    warnings: list[str] = []
    required = set(policy["required_fields"])
    forbidden = set(policy["forbidden_fields"])
    allowed_states = set(policy["allowed_seatbelt_states"])
    allowed_splits = set(policy["allowed_splits"])
    sample_ids = set()
    split_by_dimension = {
        dimension: defaultdict(set) for dimension in policy["isolation_dimensions"]
    }
    coverage: defaultdict[tuple[str, str, str], set[str]] = defaultdict(set)
    conditions = set()
    for item in records:
        sample_id = str(item.get("sample_id", "<missing>"))
        missing = required - set(item)
        if missing:
            errors.append(f"{sample_id}: missing {sorted(missing)}")
            continue
        present_forbidden = forbidden & set(item)
        if present_forbidden:
            errors.append(f"{sample_id}: duplicated/derived sample fields {sorted(present_forbidden)}")
        if sample_id in sample_ids:
            errors.append(f"duplicate sample_id: {sample_id}")
        sample_ids.add(sample_id)
        if item["split"] not in allowed_splits:
            errors.append(f"{sample_id}: invalid split {item['split']}")
        if item["seatbelt_state"] not in allowed_states:
            errors.append(f"{sample_id}: invalid seatbelt_state {item['seatbelt_state']}")
        if not _valid_yolo(item["yolo"]):
            errors.append(f"{sample_id}: invalid upper-body yolo box")
        image_path = Path(item["image_path"])
        if not image_path.is_file():
            errors.append(f"{sample_id}: image not found {image_path}")
        elif sha256_file(image_path) != item["sha256"]:
            errors.append(f"{sample_id}: image SHA mismatch")
        for dimension, values in split_by_dimension.items():
            values[str(item[dimension])].add(item["split"])
        for dimension in ("video_id", "vehicle_id", "person_id", "camera_id"):
            coverage[(item["seatbelt_state"], item["split"], dimension)].add(
                str(item[dimension])
            )
        conditions.update(str(value) for value in item["conditions"])

    overlaps = {}
    for dimension, values in split_by_dimension.items():
        overlaps[dimension] = sorted(key for key, splits in values.items() if len(splits) > 1)
        if overlaps[dimension]:
            errors.append(f"{dimension} crosses splits: {len(overlaps[dimension])}")
    minimum_report = {}
    for state, by_split in policy["minimum_independent_groups"].items():
        minimum_report[state] = {}
        for split, requirements in by_split.items():
            minimum_report[state][split] = {}
            for dimension, minimum in requirements.items():
                actual = len(coverage[(state, split, dimension)])
                minimum_report[state][split][dimension] = {
                    "actual": actual,
                    "minimum": int(minimum),
                    "pass": actual >= int(minimum),
                }
                if actual < int(minimum):
                    errors.append(
                        f"{state}/{split} has {actual} {dimension}, requires {minimum}"
                    )
    missing_conditions = sorted(set(policy["required_condition_coverage"]) - conditions)
    if missing_conditions:
        errors.append(f"missing condition coverage: {missing_conditions}")
    if not records:
        errors.append("capture manifest is empty")
    return {
        "status": "PASS" if not errors else "FAIL",
        "samples": len(records),
        "minimum_independent_groups": minimum_report,
        "condition_coverage": sorted(conditions),
        "missing_conditions": missing_conditions,
        "overlaps": overlaps,
        "errors": errors,
        "warnings": warnings,
        "policy": "PASS confirms declared diversity/isolation and file integrity, not label correctness.",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate independent V2 capture data")
    parser.add_argument("manifest", type=Path)
    parser.add_argument(
        "--policy", type=Path, default=Path("datasets/v2_capture_policy.yaml")
    )
    parser.add_argument("--output", type=Path, default=Path("reports/v2_capture_audit.json"))
    args = parser.parse_args()
    report = validate(args.manifest, args.policy)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
