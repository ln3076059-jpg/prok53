from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROLE_NAMES = (
    "driver",
    "front_passenger",
    "rear_left",
    "rear_center",
    "rear_right",
    "unknown",
)
REQUIRED = {
    "vehicle_context_truth",
    "vehicle_context_predicted",
    "cabin_localized_truth",
    "cabin_localized_predicted",
    "occupant_role_truth",
    "occupant_role_predicted",
    "phone_to_driver_truth",
    "phone_to_driver_predicted",
    "pose_available",
    "wrist_available",
    "face_keypoints_available",
}


def _binary(value: str) -> bool:
    normalized = str(value).strip().lower()
    if normalized in {"1", "true", "yes", "y"}:
        return True
    if normalized in {"0", "false", "no", "n"}:
        return False
    raise ValueError(f"invalid binary value: {value!r}")


def _accuracy(rows: list[dict], truth_key: str, predicted_key: str) -> dict:
    correct = sum(_binary(item[truth_key]) == _binary(item[predicted_key]) for item in rows)
    return {"accuracy": correct / len(rows), "correct": correct, "samples": len(rows)}


def evaluate(rows: list[dict]) -> dict:
    if not rows:
        raise ValueError("association evaluation input is empty")
    missing = REQUIRED - set(rows[0])
    if missing:
        raise ValueError(f"association evaluation is missing columns: {sorted(missing)}")

    confusion: defaultdict[str, Counter] = defaultdict(Counter)
    for item in rows:
        truth = item["occupant_role_truth"]
        predicted = item["occupant_role_predicted"]
        if truth not in ROLE_NAMES or predicted not in ROLE_NAMES:
            raise ValueError(f"invalid occupant role pair: {truth!r}, {predicted!r}")
        confusion[truth][predicted] += 1
    role_correct = sum(confusion[role][role] for role in ROLE_NAMES)

    return {
        "status": "MEASURED",
        "samples": len(rows),
        "association_metrics": {
            "vehicle_context": _accuracy(
                rows, "vehicle_context_truth", "vehicle_context_predicted"
            ),
            "cabin_localization": _accuracy(
                rows, "cabin_localized_truth", "cabin_localized_predicted"
            ),
            "occupant_role": {
                "accuracy": role_correct / len(rows),
                "correct": role_correct,
                "samples": len(rows),
                "confusion_matrix": {
                    truth: {predicted: confusion[truth][predicted] for predicted in ROLE_NAMES}
                    for truth in ROLE_NAMES
                },
            },
            "phone_to_driver": _accuracy(
                rows, "phone_to_driver_truth", "phone_to_driver_predicted"
            ),
        },
        "pose_availability": {
            "pose_rate": sum(_binary(item["pose_available"]) for item in rows) / len(rows),
            "wrist_rate": sum(_binary(item["wrist_available"]) for item in rows) / len(rows),
            "face_keypoint_rate": sum(_binary(item["face_keypoints_available"]) for item in rows)
            / len(rows),
        },
        "scientific_scope": (
            "Association and keypoint availability only; this report is not event accuracy."
        ),
    }


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate V2 vehicle/cabin/occupant/phone association and pose availability"
    )
    parser.add_argument("annotations", type=Path, nargs="?")
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2_association_evaluation.json")
    )
    args = parser.parse_args()
    report = (
        evaluate(_read(args.annotations))
        if args.annotations
        else {
            "status": "NOT_RUN",
            "reason": "independently reviewed association annotations are required",
        }
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
