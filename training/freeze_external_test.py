from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from training.common import sha256_file, stable_json_hash


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _validate_event_annotation(path: Path, sample_id: str) -> list[str]:
    try:
        events = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [f"{sample_id}: annotation is not valid JSON: {exc}"]
    if not isinstance(events, list):
        return [f"{sample_id}: annotation root must be a JSON list"]
    errors = []
    for index, event in enumerate(events):
        location = f"{sample_id}: annotation event {index}"
        if not isinstance(event, dict):
            errors.append(f"{location} must be an object")
            continue
        missing = {"event_type", "start_seconds", "end_seconds", "occupant_role"} - set(event)
        if missing:
            errors.append(f"{location} missing {sorted(missing)}")
            continue
        if event["event_type"] not in {"PHONE", "NO_SEATBELT"}:
            errors.append(f"{location} has invalid event_type {event['event_type']}")
        if event["occupant_role"] not in {
            "driver",
            "front_passenger",
            "rear_left",
            "rear_center",
            "rear_right",
            "unknown",
        }:
            errors.append(f"{location} has invalid occupant_role {event['occupant_role']}")
        try:
            start = float(event["start_seconds"])
            end = float(event["end_seconds"])
        except (TypeError, ValueError):
            errors.append(f"{location} has non-numeric timestamps")
            continue
        if start < 0 or end < start:
            errors.append(f"{location} has invalid interval {start}..{end}")
    return errors


def freeze_external_test(
    manifest_path: Path,
    development_manifest_path: Path,
    policy_path: Path,
    output_path: Path,
    identity_manifest_lock_path: Path | None = None,
) -> dict:
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    records = _read_jsonl(manifest_path)
    development = _read_jsonl(development_manifest_path)
    errors: list[str] = []
    required_fields = set(policy["required_fields"])
    disjoint_dimensions = tuple(policy["disjoint_dimensions"])
    external_values: defaultdict[str, set[str]] = defaultdict(set)
    development_values: defaultdict[str, set[str]] = defaultdict(set)
    conditions: set[str] = set()
    file_hashes: dict[str, dict[str, str]] = {}
    seen_sample_ids: set[str] = set()
    seen_unique: defaultdict[str, set[str]] = defaultdict(set)

    manifest_lock = None
    if identity_manifest_lock_path is not None:
        if not identity_manifest_lock_path.is_file():
            errors.append(f"identity manifest lock not found: {identity_manifest_lock_path}")
        else:
            manifest_lock = json.loads(identity_manifest_lock_path.read_text(encoding="utf-8"))
            if manifest_lock.get("status") != "FROZEN_IDENTITY_MANIFEST":
                errors.append("identity manifest lock status is not FROZEN_IDENTITY_MANIFEST")
            elif not manifest_lock.get("manifest_sha256"):
                errors.append("identity manifest lock is missing manifest_sha256")
            elif not manifest_lock.get("eligible_for_frozen_event_evaluation", False):
                errors.append(
                    f"identity manifest source_type '{manifest_lock.get('source_type')}' is not eligible for frozen external test evaluation"
                )
    elif policy.get("require_identity_manifest") is True:
        errors.append("external test policy requires a frozen identity manifest lock")

    for item in development:
        for dimension in disjoint_dimensions:
            if item.get(dimension) not in (None, ""):
                development_values[dimension].add(str(item[dimension]))

    for item in records:
        sample_id = str(item.get("sample_id", "<missing>"))
        missing = required_fields - set(item)
        if missing:
            errors.append(f"{sample_id}: missing {sorted(missing)}")
            continue
        if sample_id in seen_sample_ids:
            errors.append(f"duplicate sample_id: {sample_id}")
        seen_sample_ids.add(sample_id)
        for dimension in ("source_id", "camera_id", "video_id", "vehicle_id", "person_id"):
            if dimension in item and not str(item[dimension]).strip():
                errors.append(f"{sample_id}: {dimension} must not be empty")
        for dimension in ("sha256", "video_id"):
            value = str(item[dimension])
            if value in seen_unique[dimension]:
                errors.append(f"{sample_id}: duplicate external {dimension} {value}")
            seen_unique[dimension].add(value)
        if item.get("dataset_role") != policy["dataset_role"]:
            errors.append(f"{sample_id}: dataset_role must be {policy['dataset_role']}")
        if item["human_review_status"] != "APPROVED":
            errors.append(f"{sample_id}: human review is not APPROVED")
        if item["reviewer_type"] != "HUMAN":
            errors.append(f"{sample_id}: reviewer_type must be HUMAN")
        if not str(item["reviewer_id"]).strip() or not str(item["reviewed_at"]).strip():
            errors.append(f"{sample_id}: reviewer provenance is incomplete")
        else:
            try:
                reviewed_at = datetime.fromisoformat(
                    str(item["reviewed_at"]).replace("Z", "+00:00")
                )
                if reviewed_at.utcoffset() is None:
                    raise ValueError("timezone is missing")
            except ValueError:
                errors.append(
                    f"{sample_id}: reviewed_at is not a timezone-aware ISO-8601 timestamp"
                )

        video_path = Path(item["video_path"])
        annotation_path = Path(item["annotation_path"])
        if not video_path.is_file():
            errors.append(f"{sample_id}: video not found {video_path}")
        elif sha256_file(video_path) != item["sha256"]:
            errors.append(f"{sample_id}: video SHA mismatch")
        if not annotation_path.is_file():
            errors.append(f"{sample_id}: annotation not found {annotation_path}")
        elif sha256_file(annotation_path) != item["annotation_sha256"]:
            errors.append(f"{sample_id}: annotation SHA mismatch")
        else:
            errors.extend(_validate_event_annotation(annotation_path, sample_id))

        if manifest_lock is not None:
            manifest_videos = manifest_lock.get("videos", {})
            manifest_video_ids = set(manifest_lock.get("video_ids", []))
            item_vid = str(item.get("video_id", ""))
            item_sha = str(item.get("sha256", ""))
            if manifest_video_ids and item_vid not in manifest_video_ids:
                errors.append(f"{sample_id}: video_id '{item_vid}' is not declared in frozen identity manifest video_ids")
            if item_vid in manifest_videos:
                m_sha = manifest_videos[item_vid].get("sha256", "")
                if m_sha and m_sha != item_sha:
                    errors.append(
                        f"{sample_id}: video SHA in identity manifest ({m_sha}) does not match external test video SHA ({item_sha})"
                    )

        file_hashes[sample_id] = {
            "video": str(item["sha256"]),
            "annotation": str(item["annotation_sha256"]),
        }
        if not isinstance(item["conditions"], list) or not all(
            isinstance(value, str) and value.strip() for value in item["conditions"]
        ):
            errors.append(f"{sample_id}: conditions must be a list of non-empty strings")
        else:
            conditions.update(item["conditions"])
        for dimension in disjoint_dimensions:
            external_values[dimension].add(str(item[dimension]))

    overlaps = {
        dimension: sorted(external_values[dimension] & development_values[dimension])
        for dimension in disjoint_dimensions
    }
    for dimension, values in overlaps.items():
        if values:
            errors.append(f"{dimension} overlaps development data: {len(values)}")

    coverage = {}
    for dimension, minimum in policy["minimum_independent_groups"].items():
        actual = len(external_values[dimension])
        coverage[dimension] = {
            "actual": actual,
            "minimum": int(minimum),
            "pass": actual >= int(minimum),
        }
        if actual < int(minimum):
            errors.append(f"external test has {actual} {dimension}, requires {minimum}")
    missing_conditions = sorted(set(policy["required_condition_coverage"]) - conditions)
    if missing_conditions:
        errors.append(f"missing condition coverage: {missing_conditions}")
    if not records:
        errors.append("external test manifest is empty")
    if errors:
        raise ValueError("cannot freeze external test:\n- " + "\n- ".join(errors))

    if output_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen external test: {output_path}")

    frozen = {
        "schema_version": 2,
        "status": "FROZEN_EXTERNAL_TEST",
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_role": policy["dataset_role"],
        "samples": len(records),
        "manifest_sha256": stable_json_hash(records),
        "development_manifest_sha256": stable_json_hash(development),
        "dataset_sha256": stable_json_hash(file_hashes),
        "policy_sha256": sha256_file(policy_path),
        "independent_group_coverage": coverage,
        "condition_coverage": sorted(conditions),
        "video_ids": sorted(external_values["video_id"]),
        "development_overlap": overlaps,
        "human_review_status": "ALL_APPROVED",
        "scientific_claim": "INTEGRITY_AND_ISOLATION_ONLY_NOT_MODEL_ACCURACY",
    }
    if manifest_lock is not None and identity_manifest_lock_path is not None:
        frozen["identity_manifest_sha256"] = manifest_lock.get("manifest_sha256")
        frozen["evaluation_scope"] = manifest_lock.get("evaluation_scope", "FULL_SYSTEM_EVENT_EVALUATION")
        frozen["identity_manifest_lock_path"] = str(identity_manifest_lock_path.resolve())
        frozen["identity_manifest_lock"] = {
            "path": str(identity_manifest_lock_path.resolve()),
            "sha256": sha256_file(identity_manifest_lock_path),
        }
        frozen["require_identity_manifest"] = True

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(frozen, handle, indent=2, sort_keys=True)
        handle.write("\n")
    return frozen


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze a human-reviewed, external V2 test set")
    parser.add_argument("manifest", type=Path)
    parser.add_argument("development_manifest", type=Path)
    parser.add_argument(
        "--policy",
        type=Path,
        default=Path("datasets/v2_external_test_policy.yaml"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/manifests/v2_external_test_frozen.json"),
    )
    parser.add_argument(
        "--identity-manifest-lock",
        type=Path,
        default=None,
        help="Optional path to frozen identity manifest lock JSON",
    )
    args = parser.parse_args()
    report = freeze_external_test(
        args.manifest,
        args.development_manifest,
        args.policy,
        args.output,
        args.identity_manifest_lock,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
