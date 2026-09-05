from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha256_file
from training.freeze_event_ground_truth import OCCUPANT_ROLES, _parse_reviewed_at

REQUIRED_COLUMNS = {
    "video_id",
    "context_id",
    "occupant_id",
    "vehicle_id",
    "cabin_id",
    "occupant_role",
    "start_seconds",
    "end_seconds",
    "timeline_end_seconds",
    "inside_vehicle",
    "outside_vehicle_person",
    "motorcycle_flag",
    "phone_state",
    "seatbelt_state",
    "visibility",
    "conditions",
    "human_review_status",
    "reviewer_id",
    "reviewer_type",
    "reviewed_at",
    "adjudication_status",
    "notes",
}

PHONE_STATES = {
    "PHONE_USE",
    "PHONE_PRESENT_NOT_USED",
    "MOUNTED_OR_STATIC_PHONE",
    "NO_PHONE",
    "UNKNOWN",
}
SEATBELT_STATES = {
    "FASTENED",
    "UNFASTENED",
    "UNCERTAIN_OR_OCCLUDED",
    "NOT_APPLICABLE",
}
BOOLEAN_VALUES = {"true", "false"}
EPSILON_SECONDS = 1e-6


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"context truth is missing columns: {sorted(missing)}")
        return list(reader)


def _known_identity(row: dict[str, str], field: str, context_id: str) -> str:
    value = row[field].strip()
    if not value or value.upper() == "UNKNOWN":
        raise ValueError(f"{context_id}: {field} must be a known stable identity")
    return value


def freeze_context_ground_truth(
    context_path: Path,
    external_test_lock_path: Path,
    output_path: Path,
) -> dict:
    """Validate and freeze human-reviewed, full-timeline safety context."""
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen context truth: {output_path}")

    external_lock = json.loads(external_test_lock_path.read_text(encoding="utf-8"))
    if external_lock.get("status") != "FROZEN_EXTERNAL_TEST":
        raise ValueError("external test artifact is not FROZEN_EXTERNAL_TEST")
    if external_lock.get("human_review_status") != "ALL_APPROVED":
        raise ValueError("external test artifact is not fully human-approved")
    external_video_ids = set(external_lock.get("video_ids", []))
    if not external_video_ids:
        raise ValueError("external test artifact does not declare frozen video_ids")

    rows = _read_csv(context_path)
    if not rows:
        raise ValueError("context truth is empty")

    errors: list[str] = []
    context_ids: set[str] = set()
    grouped: defaultdict[tuple[str, str], list[tuple[float, float, str]]] = defaultdict(list)
    identity_signatures: defaultdict[tuple[str, str], set[tuple[str, str]]] = defaultdict(set)
    timeline_by_video: defaultdict[str, set[float]] = defaultdict(set)
    present_video_ids: set[str] = set()

    for index, row in enumerate(rows, start=2):
        context_id = row["context_id"].strip() or f"row-{index}"
        if not row["context_id"].strip():
            errors.append(f"row {index}: context_id must not be empty")
        elif context_id in context_ids:
            errors.append(f"duplicate context_id: {context_id}")
        context_ids.add(context_id)

        video_id = row["video_id"].strip()
        present_video_ids.add(video_id)
        if video_id not in external_video_ids:
            errors.append(f"{context_id}: video_id is not in the frozen external test")

        identities: dict[str, str] = {}
        for field in ("occupant_id", "vehicle_id", "cabin_id"):
            try:
                identities[field] = _known_identity(row, field, context_id)
            except ValueError as exc:
                errors.append(str(exc))

        role = row["occupant_role"].strip()
        if role not in OCCUPANT_ROLES:
            errors.append(f"{context_id}: invalid occupant_role {role!r}")
        if row["phone_state"].strip() not in PHONE_STATES:
            errors.append(f"{context_id}: invalid phone_state {row['phone_state']!r}")
        if row["seatbelt_state"].strip() not in SEATBELT_STATES:
            errors.append(f"{context_id}: invalid seatbelt_state {row['seatbelt_state']!r}")
        for field in ("inside_vehicle", "outside_vehicle_person", "motorcycle_flag"):
            if row[field].strip().lower() not in BOOLEAN_VALUES:
                errors.append(f"{context_id}: {field} must be true or false")
        if (
            row["inside_vehicle"].strip().lower() == "true"
            and row["outside_vehicle_person"].strip().lower() == "true"
        ):
            errors.append(f"{context_id}: context cannot be both inside and outside vehicle")
        for field in ("visibility", "conditions", "reviewer_id"):
            if not row[field].strip():
                errors.append(f"{context_id}: {field} must not be empty")
        if row["human_review_status"].strip() != "APPROVED":
            errors.append(f"{context_id}: human_review_status must be APPROVED")
        if row["reviewer_type"].strip() != "HUMAN":
            errors.append(f"{context_id}: reviewer_type must be HUMAN")
        if row["adjudication_status"].strip() != "FINAL":
            errors.append(f"{context_id}: adjudication_status must be FINAL")
        try:
            _parse_reviewed_at(row["reviewed_at"].strip(), context_id)
        except ValueError as exc:
            errors.append(str(exc))

        try:
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])
            timeline_end = float(row["timeline_end_seconds"])
            if not all(math.isfinite(value) for value in (start, end, timeline_end)):
                raise ValueError
            if not (0 <= start < end <= timeline_end) or timeline_end <= 0:
                raise ValueError
        except ValueError:
            errors.append(f"{context_id}: invalid half-open context interval or timeline end")
            continue

        occupant_id = identities.get("occupant_id", row["occupant_id"].strip())
        grouped[(video_id, occupant_id)].append((start, end, context_id))
        identity_signatures[(video_id, occupant_id)].add(
            (
                identities.get("vehicle_id", row["vehicle_id"].strip()),
                identities.get("cabin_id", row["cabin_id"].strip()),
            )
        )
        timeline_by_video[video_id].add(timeline_end)

    missing_videos = external_video_ids - present_video_ids
    if missing_videos:
        errors.append(
            f"context truth has no timeline coverage for frozen videos: {sorted(missing_videos)}"
        )

    for video_id, timeline_values in timeline_by_video.items():
        if len(timeline_values) != 1:
            errors.append(f"{video_id}: inconsistent timeline_end_seconds values")

    for identity, signatures in identity_signatures.items():
        if len(signatures) != 1:
            errors.append(
                f"{identity[0]}/{identity[1]}: occupant identity maps to multiple "
                "vehicle/cabin pairs"
            )

    for (video_id, occupant_id), intervals in grouped.items():
        timeline_values = timeline_by_video[video_id]
        if len(timeline_values) != 1:
            continue
        timeline_end = next(iter(timeline_values))
        cursor = 0.0
        for start, end, context_id in sorted(intervals):
            if abs(start - cursor) > EPSILON_SECONDS:
                relation = "overlap" if start < cursor else "gap"
                errors.append(
                    f"{video_id}/{occupant_id}: {relation} before {context_id}; "
                    f"expected {cursor:.9f}, got {start:.9f}"
                )
            cursor = max(cursor, end)
        if abs(cursor - timeline_end) > EPSILON_SECONDS:
            errors.append(
                f"{video_id}/{occupant_id}: incomplete timeline coverage; "
                f"ends at {cursor:.9f}, expected {timeline_end:.9f}"
            )

    if errors:
        raise ValueError("cannot freeze context ground truth:\n- " + "\n- ".join(errors))

    timeline_seconds = {
        video_id: next(iter(values)) for video_id, values in sorted(timeline_by_video.items())
    }
    frozen = {
        "schema_version": 1,
        "status": "FROZEN_CONTEXT_GROUND_TRUTH",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "context_csv": {
            "path": str(context_path.resolve()),
            "sha256": sha256_file(context_path),
        },
        "external_test_lock": {
            "path": str(external_test_lock_path.resolve()),
            "sha256": sha256_file(external_test_lock_path),
        },
        "rows": len(rows),
        "video_ids": sorted(present_video_ids),
        "occupant_entities": len(grouped),
        "timeline_seconds": timeline_seconds,
        "total_timeline_seconds": sum(timeline_seconds.values()),
        "coverage_status": "FULL_TIMELINE_NO_GAPS_OR_OVERLAPS",
        "identity_status": "KNOWN_STABLE_OCCUPANT_VEHICLE_CABIN",
        "human_review_status": "ALL_APPROVED",
        "adjudication_status": "ALL_FINAL",
        "scientific_claim": "FROZEN_HUMAN_CONTEXT_NOT_MODEL_ACCURACY",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze independently reviewed full-timeline V2 safety context truth"
    )
    parser.add_argument("context", type=Path)
    parser.add_argument("external_test_lock", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/manifests/v2_context_truth_frozen.json"),
    )
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_context_ground_truth(args.context, args.external_test_lock, args.output),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
