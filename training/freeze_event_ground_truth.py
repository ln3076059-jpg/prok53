from __future__ import annotations

import argparse
import csv
import json
import math
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha256_file
from training.identity_contract import validate_identity_contract

REQUIRED_COLUMNS = {
    "video_id",
    "event_id",
    "event_type",
    "start_seconds",
    "end_seconds",
    "occupant_id",
    "vehicle_id",
    "cabin_id",
    "occupant_role",
    "inside_vehicle",
    "outside_vehicle_person",
    "motorcycle_flag",
    "label",
    "visibility",
    "conditions",
    "human_review_status",
    "reviewer_id",
    "reviewer_type",
    "reviewed_at",
    "adjudication_status",
    "notes",
    "identity_manifest_sha256",
}
EVENT_TYPES = {"PHONE", "NO_SEATBELT"}
EVENT_LABELS = {
    "PHONE": {
        "PHONE_USE",
        "PHONE_PRESENT_NOT_USED",
        "MOUNTED_OR_STATIC_PHONE",
        "NO_PHONE",
        "UNKNOWN",
    },
    "NO_SEATBELT": {
        "FASTENED",
        "UNFASTENED",
        "UNCERTAIN_OR_OCCLUDED",
        "NOT_APPLICABLE",
    },
}
OCCUPANT_ROLES = {
    "driver",
    "front_passenger",
    "rear_left",
    "rear_center",
    "rear_right",
    "unknown",
}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        missing = REQUIRED_COLUMNS - set(reader.fieldnames or [])
        if missing:
            raise ValueError(f"event truth is missing columns: {sorted(missing)}")
        return list(reader)


def _parse_reviewed_at(value: str, event_id: str) -> None:
    try:
        reviewed_at = datetime.fromisoformat(value.replace("Z", "+00:00"))
        if reviewed_at.utcoffset() is None:
            raise ValueError("timezone is missing")
    except ValueError as exc:
        raise ValueError(
            f"{event_id}: reviewed_at must be a timezone-aware ISO-8601 timestamp"
        ) from exc


def freeze_event_ground_truth(
    truth_path: Path,
    external_test_lock_path: Path,
    output_path: Path,
    identity_manifest_lock_path: Path | None = None,
) -> dict:
    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen event truth: {output_path}")
    external_lock = json.loads(external_test_lock_path.read_text(encoding="utf-8"))
    if external_lock.get("status") != "FROZEN_EXTERNAL_TEST":
        raise ValueError("external test artifact is not FROZEN_EXTERNAL_TEST")
    if external_lock.get("human_review_status") != "ALL_APPROVED":
        raise ValueError("external test artifact is not fully human-approved")

    if identity_manifest_lock_path is None and "identity_manifest_lock_path" in external_lock:
        identity_manifest_lock_path = Path(external_lock["identity_manifest_lock_path"])

    if identity_manifest_lock_path is None:
        raise ValueError("freezing event ground truth requires a frozen identity manifest lock")

    if not identity_manifest_lock_path.is_file():
        raise ValueError(f"identity manifest lock not found: {identity_manifest_lock_path}")
    manifest_lock = json.loads(identity_manifest_lock_path.read_text(encoding="utf-8"))
    if manifest_lock.get("status") != "FROZEN_IDENTITY_MANIFEST":
        raise ValueError("identity manifest lock status is not FROZEN_IDENTITY_MANIFEST")
    manifest_sha = manifest_lock.get("manifest_sha256")
    if not manifest_sha:
        raise ValueError("identity manifest lock is missing manifest_sha256")
    if not manifest_lock.get("eligible_for_frozen_event_evaluation", False):
        raise ValueError(
            f"identity manifest source_type '{manifest_lock.get('source_type')}' is not eligible for frozen event evaluation "
            "(legacy prediction/candidate manifests are disallowed in final frozen evaluation)"
        )
    if "identity_manifest_sha256" in external_lock:
        if manifest_sha != external_lock["identity_manifest_sha256"]:
            raise ValueError("identity manifest SHA256 does not match external-test lock requirement")
    manifest_proven_identities: set[tuple[str, str, str, str]] = set()
    for entity in manifest_lock.get("proven_identities", []):
        manifest_proven_identities.add(
            (
                entity["video_id"],
                entity["vehicle_id"],
                entity["cabin_id"],
                entity["occupant_id"],
            )
        )

    rows = _read_csv(truth_path)
    if not rows:
        raise ValueError("event truth is empty")
    external_video_ids = set(external_lock.get("video_ids", []))
    if not external_video_ids:
        raise ValueError("external test artifact does not declare frozen video_ids")

    errors: list[str] = []
    event_ids: set[str] = set()
    video_ids: set[str] = set()
    event_type_counts = {event_type: 0 for event_type in sorted(EVENT_TYPES)}
    for index, row in enumerate(rows, start=2):
        event_id = row["event_id"].strip() or f"row-{index}"
        if not row["event_id"].strip():
            errors.append(f"row {index}: event_id must not be empty")
        elif event_id in event_ids:
            errors.append(f"duplicate event_id: {event_id}")
        event_ids.add(event_id)

        video_id = row["video_id"].strip()
        video_ids.add(video_id)
        if video_id not in external_video_ids:
            errors.append(f"{event_id}: video_id is not in the frozen external test")
        event_type = row["event_type"].strip()
        if event_type not in EVENT_TYPES:
            errors.append(f"{event_id}: invalid event_type {event_type!r}")
        else:
            event_type_counts[event_type] += 1
            if row["label"].strip() not in EVENT_LABELS[event_type]:
                errors.append(f"{event_id}: invalid label {row['label']!r} for {event_type}")
        if row["occupant_role"].strip() not in OCCUPANT_ROLES:
            errors.append(f"{event_id}: invalid occupant_role {row['occupant_role']!r}")
        for field in ("occupant_id", "vehicle_id", "cabin_id"):
            value = row[field].strip()
            if not value or value.upper() == "UNKNOWN":
                errors.append(f"{event_id}: {field} must be a known stable identity")
        for err in validate_identity_contract(
            video_id,
            row["vehicle_id"].strip(),
            row["cabin_id"].strip(),
            row["occupant_id"].strip(),
        ):
            errors.append(f"{event_id}: {err}")

        if manifest_lock is not None:
            row_entity = (
                video_id,
                row["vehicle_id"].strip(),
                row["cabin_id"].strip(),
                row["occupant_id"].strip(),
            )
            if row_entity not in manifest_proven_identities:
                errors.append(
                    f"{event_id}: identity {row_entity} is not in the frozen identity manifest"
                )

        for field in (
            "inside_vehicle",
            "outside_vehicle_person",
            "motorcycle_flag",
            "label",
            "visibility",
            "conditions",
            "reviewer_id",
            "identity_manifest_sha256",
        ):
            if not row[field].strip():
                errors.append(f"{event_id}: {field} must not be empty; use UNKNOWN if unavailable")
        row_manifest_sha = row["identity_manifest_sha256"].strip()
        if row_manifest_sha and row_manifest_sha != manifest_sha:
            errors.append(
                f"{event_id}: annotation identity_manifest_sha256 ({row_manifest_sha}) "
                f"does not match frozen manifest lock SHA ({manifest_sha})"
            )
        for field in ("inside_vehicle", "outside_vehicle_person", "motorcycle_flag"):
            if row[field].strip().lower() not in {"true", "false"}:
                errors.append(f"{event_id}: {field} must be true or false")
        if (
            row["inside_vehicle"].strip().lower() == "true"
            and row["outside_vehicle_person"].strip().lower() == "true"
        ):
            errors.append(f"{event_id}: event cannot be both inside and outside vehicle")
        if row["human_review_status"].strip() != "APPROVED":
            errors.append(f"{event_id}: human_review_status must be APPROVED")
        if row["reviewer_type"].strip() != "HUMAN":
            errors.append(f"{event_id}: reviewer_type must be HUMAN")
        if row["adjudication_status"].strip() != "FINAL":
            errors.append(f"{event_id}: adjudication_status must be FINAL")
        try:
            _parse_reviewed_at(row["reviewed_at"].strip(), event_id)
        except ValueError as exc:
            errors.append(str(exc))
        try:
            start = float(row["start_seconds"])
            end = float(row["end_seconds"])
            if not math.isfinite(start) or not math.isfinite(end) or start < 0 or end < start:
                raise ValueError
        except ValueError:
            errors.append(f"{event_id}: invalid event interval")

    if errors:
        raise ValueError("cannot freeze event ground truth:\n- " + "\n- ".join(errors))

    frozen = {
        "schema_version": 2,
        "status": "FROZEN_EVENT_GROUND_TRUTH",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "truth_csv": {"path": str(truth_path.resolve()), "sha256": sha256_file(truth_path)},
        "external_test_lock": {
            "path": str(external_test_lock_path.resolve()),
            "sha256": sha256_file(external_test_lock_path),
        },
        "rows": len(rows),
        "video_ids": sorted(video_ids),
        "event_type_counts": event_type_counts,
        "human_review_status": "ALL_APPROVED",
        "adjudication_status": "ALL_FINAL",
        "scientific_claim": "FROZEN_HUMAN_GROUND_TRUTH_NOT_MODEL_ACCURACY",
    }
    frozen["identity_manifest_sha256"] = manifest_sha
    frozen["evaluation_scope"] = manifest_lock.get("evaluation_scope", "FULL_SYSTEM_EVENT_EVALUATION")
    frozen["identity_manifest_lock"] = {
        "path": str(identity_manifest_lock_path.resolve()),
        "sha256": sha256_file(identity_manifest_lock_path),
        "manifest_sha256": manifest_sha,
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze independently reviewed V2 event truth")
    parser.add_argument("truth", type=Path)
    parser.add_argument("external_test_lock", type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/manifests/v2_event_truth_frozen.json"),
    )
    parser.add_argument(
        "--identity-manifest-lock",
        type=Path,
        default=None,
        help="Optional path to frozen identity manifest lock JSON",
    )
    args = parser.parse_args()
    print(
        json.dumps(
            freeze_event_ground_truth(
                args.truth,
                args.external_test_lock,
                args.output,
                args.identity_manifest_lock,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
