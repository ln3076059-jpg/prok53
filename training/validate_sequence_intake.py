"""Check declared physical lineage and evidence before admitting untouched sequences."""
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path

from training.common import sha256_file, stable_json_hash
from training.freeze_development_lineage import verify_development_lineage

DISJOINT_DIMENSIONS = (
    "sha256", "source_id", "camera_id", "video_id", "capture_session_id",
    "physical_vehicle_group_id", "person_group_ids",
)
UNKNOWN_VALUES = {"unknown", "not_provable", "pending", "none", "null", "n/a", "unreviewed"}


def dimension_values(item: dict, dimension: str) -> set[str]:
    value = item.get(dimension)
    if dimension == "person_group_ids":
        parts = value.split(";") if isinstance(value, str) else value
        if not isinstance(parts, list) or not parts:
            raise ValueError("person_group_ids must be a non-empty list or semicolon-separated IDs")
    else:
        parts = [value]
    result = set()
    for part in parts:
        if not isinstance(part, str) or not part.strip():
            raise ValueError(f"{dimension} requires non-empty string IDs")
        identifier = part.strip()
        if identifier.lower() in UNKNOWN_VALUES:
            raise ValueError(f"{dimension} is not proven: {identifier}")
        if dimension in {"physical_vehicle_group_id", "person_group_ids", "capture_session_id"}:
            if identifier.lower().startswith("video:"):
                raise ValueError(f"{dimension} must identify a physical group, not a video namespace")
        if dimension == "sha256":
            if not re.fullmatch(r"[0-9a-fA-F]{64}", identifier):
                raise ValueError("sha256 must be a valid SHA-256 digest")
            identifier = identifier.lower()
        result.add(identifier)
    return result


def validate_sequence_intake(
    holdout: list[dict], development: list[dict], *,
    development_manifest_path: Path | None = None,
    development_lineage_lock_path: Path | None = None,
    require_development_lineage_lock: bool = False,
) -> dict:
    errors: list[str] = []
    lineage_binding = None
    if require_development_lineage_lock or development_lineage_lock_path is not None:
        if development_manifest_path is None or development_lineage_lock_path is None:
            errors.append("official intake requires a frozen development lineage lock")
        else:
            try:
                lineage_binding = verify_development_lineage(
                    development_manifest_path, development_lineage_lock_path,
                )
                if stable_json_hash(development) != stable_json_hash(read_intake(development_manifest_path)):
                    raise ValueError("development rows do not match the locked input file")
            except (OSError, ValueError) as exc:
                errors.append(f"development lineage: {exc}")
    values = {
        role: {dimension: set() for dimension in DISJOINT_DIMENSIONS}
        for role in ("holdout", "development")
    }
    evidence: dict[str, dict] = {}
    for role, records in (("holdout", holdout), ("development", development)):
        if not records:
            errors.append(f"{role} manifest is empty; independence cannot be established")
        for index, item in enumerate(records):
            location = f"{role} row {index + 1}"
            if not isinstance(item, dict):
                errors.append(f"{location} must be an object")
                continue
            for dimension in DISJOINT_DIMENSIONS:
                try:
                    values[role][dimension].update(dimension_values(item, dimension))
                except ValueError as exc:
                    errors.append(f"{location}: {exc}")
            if role == "development":
                continue
            if item.get("proposed_role") != "NEW_UNTOUCHED_HOLDOUT":
                errors.append(f"{location}: proposed_role must be NEW_UNTOUCHED_HOLDOUT")
            seen = item.get("model_predictions_seen")
            # CSV has strings; JSON must use a boolean. Zero/null/UNKNOWN are not false.
            if seen is not False and seen != "false":
                errors.append(f"{location}: model_predictions_seen must be explicitly false")
            if item.get("prior_usage") != "NEVER_USED":
                errors.append(f"{location}: prior_usage must be NEVER_USED with evidence")
            for kind in ("rights", "independence"):
                path_value = item.get(f"{kind}_evidence_path")
                digest = item.get(f"{kind}_evidence_sha256")
                if not isinstance(path_value, str) or not path_value.strip():
                    errors.append(f"{location}: {kind} evidence path is missing")
                    continue
                path = Path(path_value)
                if not path.is_file() or path.stat().st_size == 0:
                    errors.append(f"{location}: {kind} evidence must be an existing non-empty file")
                    continue
                actual = sha256_file(path)
                if not isinstance(digest, str) or actual != digest.lower():
                    errors.append(f"{location}: {kind} evidence SHA256 mismatch or missing")
                    continue
                evidence[f"{index + 1}:{kind}"] = {
                    "path": str(path.resolve()), "sha256": actual,
                }
            video_value = item.get("video_path")
            video = Path(video_value) if isinstance(video_value, str) and video_value else None
            if video is None or not video.is_file() or video.stat().st_size == 0:
                errors.append(f"{location}: video must be an existing non-empty file")
            elif sha256_file(video) != str(item.get("sha256", "")).lower():
                errors.append(f"{location}: video SHA256 mismatch")
    overlaps = {
        dimension: sorted(values["holdout"][dimension] & values["development"][dimension])
        for dimension in DISJOINT_DIMENSIONS
    }
    for dimension, shared in overlaps.items():
        if shared:
            errors.append(f"{dimension} overlaps development data: {len(shared)}")
    return {
        "status": "NOT_READY_FOR_UNTOUCHED_FREEZE" if errors else "SEQUENCE_INTAKE_CHECKS_PASSED",
        "errors": errors,
        "development_overlap": overlaps,
        "evidence_files": evidence,
        "development_lineage_lock": lineage_binding,
        "scientific_claim": "DECLARED_LINEAGE_AND_EVIDENCE_INTEGRITY_ONLY",
    }


def read_intake(path: Path) -> list[dict]:
    if path.suffix.lower() == ".csv":
        with path.open(encoding="utf-8-sig", newline="") as handle:
            return list(csv.DictReader(handle))
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("holdout", type=Path, help="Holdout-only CSV or JSONL intake")
    parser.add_argument("development", type=Path, help="Complete development lineage CSV or JSONL")
    parser.add_argument("--output", type=Path, help="New path for the validation report")
    parser.add_argument("--development-lineage-lock", type=Path)
    args = parser.parse_args()
    report = validate_sequence_intake(
        read_intake(args.holdout), read_intake(args.development),
        development_manifest_path=args.development,
        development_lineage_lock_path=args.development_lineage_lock,
        require_development_lineage_lock=True,
    )
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
            handle.write("\n")
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if not report["errors"] else 1)


if __name__ == "__main__":
    main()
