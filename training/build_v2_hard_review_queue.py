from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

SCENARIO_BY_SOURCE_CLASS = {
    "Cigarette": "SMALL_HANDHELD_RECTANGULAR_CONFUSER",
    "Neutral": "NEUTRAL_DRIVER_HARD_NEGATIVE",
    "Distraction": "NON_PHONE_DRIVER_ACTION",
    "drinking": "BOTTLE_OR_CUP_HANDHELD_CONFUSER",
    "yawning": "HAND_NEAR_FACE_CONFUSER",
}

REQUIRED_NEW_CAPTURE_SCENARIOS = [
    "PHONE_MOUNTED_OR_STATIC",
    "PASSENGER_USING_PHONE",
    "PERSON_OUTSIDE_VEHICLE_HOLDING_PHONE",
    "DRIVER_SCRATCHING_HEAD",
    "DRIVER_ADJUSTING_MIRROR",
    "DRIVER_COVERING_FACE",
    "DRIVER_HOLDING_WALLET_OR_RECTANGULAR_OBJECT",
    "UNFASTENED_INDEPENDENT_VEHICLE_PERSON_CAMERA",
    "UNCERTAIN_OCCLUDED_BELT",
    "NIGHT_GLARE_REFLECTION_MOTION_COMPRESSION",
]


def _read(path: Path) -> list[dict]:
    return json.loads(path.read_text(encoding="utf-8"))


def build(inputs: list[Path], output: Path, per_group: int = 1) -> dict:
    if per_group < 1:
        raise ValueError("per_group must be positive")
    candidates: defaultdict[tuple[str, str], list[dict]] = defaultdict(list)
    source_totals: defaultdict[str, int] = defaultdict(int)
    for path in inputs:
        for item in _read(path):
            scenarios = sorted(
                {
                    SCENARIO_BY_SOURCE_CLASS[annotation["source_class_name"]]
                    for annotation in item.get("excluded_source_annotations", [])
                    if annotation.get("source_class_name") in SCENARIO_BY_SOURCE_CLASS
                }
            )
            for scenario in scenarios:
                key = (scenario, item["source_group_id"])
                candidates[key].append(item)
                source_totals[scenario] += 1
    selected = []
    scenario_groups: defaultdict[str, set[str]] = defaultdict(set)
    for (scenario, group_id), items in sorted(candidates.items()):
        for item in sorted(items, key=lambda value: value["sample_id"])[:per_group]:
            original_path = Path(item["image_path"])
            image_path = original_path
            if not image_path.is_file():
                relocated = (
                    Path("datasets/incoming") / original_path.parent.name / original_path.name
                )
                if relocated.is_file():
                    image_path = relocated.resolve()
            selected.append(
                {
                    "sample_id": item["sample_id"],
                    "source_id": item["source_id"],
                    "source_asset_id": item["source_asset_id"],
                    "source_group_id": group_id,
                    "image_path": str(image_path),
                    "license": item.get("license"),
                    "scenario": scenario,
                    "review_status": "PENDING",
                    "allowed_decisions": [
                        "TRUE_PHONE_HANDHELD",
                        "PHONE_MOUNTED_OR_STATIC",
                        "NON_PHONE_HARD_NEGATIVE",
                        "OUTSIDE_VEHICLE",
                        "PASSENGER",
                        "UNCERTAIN",
                        "REJECT",
                    ],
                    "automatic_training_label": None,
                    "review_tasks": [
                        "VEHICLE_CONTEXT_REVIEW",
                        "OCCUPANT_ROLE_REVIEW",
                        "PHYSICAL_PHONE_REBOX_IF_VISIBLE",
                        "HARD_NEGATIVE_DECISION",
                    ],
                }
            )
            scenario_groups[scenario].add(group_id)
    report = {
        "status": "HUMAN_REVIEW_REQUIRED",
        "selection_policy": "AT_MOST_ONE_FRAME_PER_SOURCE_GROUP_AND_SCENARIO",
        "selected": selected,
        "selected_samples": len(selected),
        "scenario_counts_before_grouping": dict(sorted(source_totals.items())),
        "independent_group_counts": {
            key: len(value) for key, value in sorted(scenario_groups.items())
        },
        "required_new_capture_scenarios": REQUIRED_NEW_CAPTURE_SCENARIOS,
        "policy": "No candidate is a detector negative or event label until explicitly reviewed.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build group-diverse V2 hard-example review queue")
    parser.add_argument(
        "--inputs",
        type=Path,
        nargs="+",
        default=[
            Path("datasets/manifests/review_queue_dms.json"),
            Path("datasets/manifests/review_queue_mendeley_driver_risk_v1.json"),
            Path("datasets/manifests/review_queue_anywaylabs_synthetic_dms.json"),
        ],
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/manifests/v2_hard_review_queue.json")
    )
    parser.add_argument("--per-group", type=int, default=1)
    args = parser.parse_args()
    report = build(args.inputs, args.output, args.per_group)
    print(
        json.dumps(
            {key: value for key, value in report.items() if key != "selected"}, indent=2
        )
    )


if __name__ == "__main__":
    main()
