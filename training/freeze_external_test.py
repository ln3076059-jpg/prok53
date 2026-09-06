from __future__ import annotations

import argparse
import json
from collections import defaultdict
from datetime import UTC, datetime
from pathlib import Path

import yaml

from training.common import sha256_file, stable_json_hash
from training.validate_sequence_intake import dimension_values, validate_sequence_intake
from training.validate_event_sequence_annotations import load_schema, validate_annotation
from training.sequence_truth_binding import sequence_bindings as canonical_sequence_bindings


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


def _validate_sequence_annotations(
    item: dict, manifest_lock: dict | None, *, require_review_evidence: bool = False,
) -> tuple[list[str], list[dict]]:
    """Bind every cabin's canonical sequence source; no second event-list truth is needed."""
    errors: list[str] = []
    bindings: list[dict] = []
    sample_id = item["sample_id"]
    extra = item.get("additional_sequence_annotations", [])
    if not isinstance(extra, list):
        return [f"{sample_id}: additional_sequence_annotations must be a list"], []
    sources = [{"path": item["annotation_path"], "sha256": item["annotation_sha256"]}, *extra]
    if manifest_lock is None:
        return [f"{sample_id}: canonical sequences require a frozen identity manifest"], []
    schema = load_schema(
        Path(__file__).resolve().parents[1] / "datasets/schemas/v2_event_sequence_annotation.schema.json"
    )
    video_id = item["video_id"]
    metadata = manifest_lock.get("videos", {}).get(video_id, {})
    expected = {
        (p["vehicle_id"], p["cabin_id"], p["occupant_id"])
        for p in manifest_lock.get("proven_identities", []) if p.get("video_id") == video_id
    }
    vehicles = {identity[0] for identity in expected}
    physical_groups = item.get("vehicle_physical_groups")
    if len(vehicles) > 1 or physical_groups is not None:
        if not isinstance(physical_groups, dict) or set(physical_groups) != vehicles:
            errors.append(f"{sample_id}: vehicle_physical_groups must map every canonical vehicle")
    observed: set[tuple[str, str, str]] = set()
    seen_paths: set[Path] = set()
    for record in sources:
        if not isinstance(record, dict):
            errors.append(f"{sample_id}: sequence reference must be a path/SHA256 object")
            continue
        path = Path(str(record.get("path", "")))
        if path.resolve() in seen_paths:
            errors.append(f"{sample_id}: duplicate sequence annotation path")
            continue
        seen_paths.add(path.resolve())
        if not path.is_file() or record.get("sha256") != sha256_file(path):
            errors.append(f"{sample_id}: sequence annotation missing or SHA256 mismatch: {path}")
            continue
        try:
            annotation = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError) as exc:
            errors.append(f"{sample_id}: invalid sequence JSON: {exc}")
            continue
        if not isinstance(annotation, dict):
            errors.append(f"{sample_id}: canonical sequence annotation must be a JSON object")
            continue
        annotation_errors = validate_annotation(annotation, schema, require_review_evidence=require_review_evidence)
        if annotation_errors:
            errors.extend(f"{sample_id}: {error}" for error in annotation_errors)
            continue
        for field, value in (
            ("video_id", video_id), ("video_sha256", item["sha256"]),
            ("source_id", item["source_id"]), ("camera_id", item["camera_id"]),
            ("fps", metadata.get("fps")), ("frame_count", metadata.get("frame_count")),
        ):
            if value is None or annotation.get(field) != value:
                errors.append(f"{sample_id}: sequence {field} does not match frozen video/manifest")
        review = annotation["review_provenance"]
        if review.get("reviewer_type") != "HUMAN" or review.get("adjudication_status") != "FINAL":
            errors.append(f"{sample_id}: sequence requires HUMAN review and FINAL adjudication")
        if str(review.get("reviewer_id", "")).strip().lower() in {
            "", "unknown", "none", "placeholder", "human-reviewer-1",
        }:
            errors.append(f"{sample_id}: sequence requires a real reviewer_id")
        if review.get("identity_manifest_sha256") != manifest_lock["manifest_sha256"]:
            errors.append(f"{sample_id}: sequence identity manifest SHA256 mismatch")
        try:
            reviewed = datetime.fromisoformat(str(review.get("reviewed_at", "")).replace("Z", "+00:00"))
            if reviewed.utcoffset() is None:
                raise ValueError("missing timezone")
        except ValueError:
            errors.append(f"{sample_id}: sequence reviewed_at must be timezone-aware")
        for occupant in annotation["occupants"]:
            identity = (annotation["vehicle_id"], annotation["cabin_id"], occupant["occupant_id"])
            if identity in observed:
                errors.append(f"{sample_id}: occupant repeated across sequence annotations")
            observed.add(identity)
        bindings.append({"path": str(path.resolve()), "sha256": sha256_file(path)})
    if not expected or observed != expected:
        errors.append(f"{sample_id}: sequences must cover exactly the frozen video's proven occupants")
    return errors, bindings


def freeze_external_test(
    manifest_path: Path,
    development_manifest_path: Path,
    policy_path: Path,
    output_path: Path,
    identity_manifest_lock_path: Path | None = None,
    development_lineage_lock_path: Path | None = None,
) -> dict:
    policy = yaml.safe_load(policy_path.read_text(encoding="utf-8"))
    records = _read_jsonl(manifest_path)
    development = _read_jsonl(development_manifest_path)
    errors: list[str] = []
    intake_validation = None
    if policy.get("require_sequence_intake") is True:
        intake_validation = validate_sequence_intake(
            records, development,
            development_manifest_path=development_manifest_path,
            development_lineage_lock_path=development_lineage_lock_path,
            require_development_lineage_lock=policy.get("require_development_lineage_lock") is True,
        )
        errors.extend(intake_validation["errors"])
    required_fields = set(policy["required_fields"])
    disjoint_dimensions = tuple(policy["disjoint_dimensions"])
    external_values: defaultdict[str, set[str]] = defaultdict(set)
    development_values: defaultdict[str, set[str]] = defaultdict(set)
    conditions: set[str] = set()
    file_hashes: dict[str, dict[str, str]] = {}
    seen_sample_ids: set[str] = set()
    seen_unique: defaultdict[str, set[str]] = defaultdict(set)
    sequence_bindings: dict[str, list[dict]] = {}

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
    elif (
        policy.get("require_identity_manifest") is True
        or policy.get("require_full_system_evaluation") is True
    ):
        errors.append("external test policy requires a frozen identity manifest lock")

    for item in development:
        for dimension in disjoint_dimensions:
            if item.get(dimension) not in (None, ""):
                try:
                    development_values[dimension].update(dimension_values(item, dimension))
                except ValueError as exc:
                    errors.append(f"development: {exc}")

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
        elif policy.get("require_canonical_sequence_annotations") is not True:
            errors.extend(_validate_event_annotation(annotation_path, sample_id))

        if policy.get("require_canonical_sequence_annotations") is True:
            sequence_errors, bindings = _validate_sequence_annotations(
                item, manifest_lock,
                require_review_evidence=policy.get("require_sequence_review_evidence") is True,
            )
            errors.extend(sequence_errors)
            sequence_bindings[sample_id] = bindings

        if manifest_lock is not None:
            manifest_videos = manifest_lock.get("videos", {})
            manifest_video_ids = set(manifest_lock.get("video_ids", []))
            item_vid = str(item.get("video_id", ""))
            item_sha = str(item.get("sha256", "")).strip().lower()
            if not manifest_video_ids or item_vid not in manifest_video_ids:
                errors.append(f"{sample_id}: video_id '{item_vid}' is not declared in frozen identity manifest video_ids")
            if item_vid not in manifest_videos:
                errors.append(f"{sample_id}: video_id '{item_vid}' is missing from frozen identity manifest videos mapping")
            else:
                m_sha = str(manifest_videos[item_vid].get("sha256", "")).strip()
                if not m_sha or len(m_sha) != 64:
                    errors.append(
                        f"{sample_id}: video SHA in identity manifest for '{item_vid}' is missing or not 64 characters ({m_sha!r})"
                    )
                elif m_sha.lower() != item_sha.lower():
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
            try:
                external_values[dimension].update(dimension_values(item, dimension))
            except ValueError as exc:
                errors.append(f"{sample_id}: {exc}")

    if manifest_lock is not None:
        external_vids = set(external_values["video_id"])
        manifest_vids = set(manifest_lock.get("video_ids", []))
        extra_manifest_vids = sorted(manifest_vids - external_vids)
        if extra_manifest_vids:
            errors.append(
                f"frozen identity manifest covers videos not present in external test manifest: {extra_manifest_vids}"
            )
        if policy.get("require_full_system_evaluation") is True:
            manifest_scope = manifest_lock.get("evaluation_scope")
            if manifest_scope != "FULL_SYSTEM_EVENT_EVALUATION":
                errors.append(
                    f"external test policy requires FULL_SYSTEM_EVENT_EVALUATION but identity manifest has '{manifest_scope}'"
                )

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

    if intake_validation is not None:
        frozen["sequence_intake_validation"] = intake_validation
        if intake_validation.get("development_lineage_lock"):
            frozen["development_lineage_lock"] = intake_validation["development_lineage_lock"]
    if sequence_bindings:
        frozen["sequence_annotations"] = sequence_bindings
        frozen["require_canonical_sequence_annotations"] = True
        frozen["require_sequence_review_evidence"] = policy.get("require_sequence_review_evidence") is True
        frozen["source_sequence_set_sha256"] = stable_json_hash(canonical_sequence_bindings(frozen))

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
    parser.add_argument(
        "--development-lineage-lock", type=Path,
        help="Frozen human-reviewed completeness attestation for the exact development manifest",
    )
    args = parser.parse_args()
    report = freeze_external_test(
        args.manifest,
        args.development_manifest,
        args.policy,
        args.output,
        args.identity_manifest_lock,
        args.development_lineage_lock,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
