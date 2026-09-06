"""Freeze a human attestation of the complete development/previous-use inventory."""
from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha256_file

COVERED_USAGE = {
    "TRAIN", "VALIDATION", "THRESHOLD_CALIBRATION", "TEMPORAL_CALIBRATION",
    "FUSION_TRAINING", "PREVIOUSLY_USED",
}


def _review(lineage_path: Path, review_path: Path) -> tuple[dict, dict]:
    from training.validate_sequence_intake import DISJOINT_DIMENSIONS, dimension_values, read_intake

    rows = read_intake(lineage_path)
    if not rows:
        raise ValueError("development lineage is empty")
    for row in rows:
        for dimension in DISJOINT_DIMENSIONS:
            dimension_values(row, dimension)
    review = json.loads(review_path.read_text(encoding="utf-8"))
    for field, expected in (
        ("human_review_status", "APPROVED"), ("reviewer_type", "HUMAN"),
        ("adjudication_status", "FINAL"), ("completeness_status", "COMPLETE"),
    ):
        if review.get(field) != expected:
            raise ValueError(f"development lineage review requires {field}={expected}")
    reviewer = review.get("reviewer_id")
    if not isinstance(reviewer, str) or reviewer.strip().lower() in {
        "", "unknown", "none", "placeholder", "human-reviewer-1",
    }:
        raise ValueError("development lineage review requires a real reviewer_id")
    try:
        stamp = datetime.fromisoformat(str(review.get("reviewed_at", "")).replace("Z", "+00:00"))
        if stamp.utcoffset() is None:
            raise ValueError("timezone missing")
    except ValueError as exc:
        raise ValueError("development lineage reviewed_at must be timezone-aware") from exc
    covered = review.get("covered_usage")
    if not isinstance(covered, list) or not all(isinstance(v, str) for v in covered):
        raise ValueError("development lineage covered_usage must list all usage categories")
    if not COVERED_USAGE.issubset(covered):
        raise ValueError("development lineage review does not cover all usage categories")
    if review.get("lineage_sha256") != sha256_file(lineage_path):
        raise ValueError("development lineage SHA256 does not match human review")
    evidence = review.get("completeness_evidence", {})
    path = Path(str(evidence.get("path", "")))
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError("development lineage completeness evidence is missing or empty")
    if evidence.get("sha256") != sha256_file(path):
        raise ValueError("development lineage completeness evidence SHA256 mismatch")
    return review, {"path": str(path.resolve()), "sha256": sha256_file(path)}


def freeze_development_lineage(lineage_path: Path, review_path: Path, output_path: Path) -> dict:
    if output_path.exists():
        raise FileExistsError("refusing to overwrite frozen development lineage")
    review, evidence = _review(lineage_path, review_path)
    lock = {
        "status": "FROZEN_DEVELOPMENT_LINEAGE",
        "locked_at": datetime.now(UTC).isoformat(),
        "lineage": {"path": str(lineage_path.resolve()), "sha256": sha256_file(lineage_path)},
        "review_source": {"path": str(review_path.resolve()), "sha256": sha256_file(review_path)},
        "review": review,
        "completeness_evidence": evidence,
        "scientific_claim": "HUMAN_ATTESTED_COMPLETENESS_AND_INTEGRITY_ONLY",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("x", encoding="utf-8") as handle:
        json.dump(lock, handle, indent=2)
        handle.write("\n")
    return lock


def verify_development_lineage(lineage_path: Path, lock_path: Path) -> dict:
    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    if lock.get("status") != "FROZEN_DEVELOPMENT_LINEAGE":
        raise ValueError("development lineage lock is not FROZEN_DEVELOPMENT_LINEAGE")
    if lock.get("lineage") != {
        "path": str(lineage_path.resolve()), "sha256": sha256_file(lineage_path),
    }:
        raise ValueError("development lineage lock does not bind the exact input path/SHA256")
    source = lock.get("review_source", {})
    path = Path(str(source.get("path", "")))
    if not path.is_file() or source.get("sha256") != sha256_file(path):
        raise ValueError("development lineage review source missing or SHA256 mismatch")
    review, evidence = _review(lineage_path, path)
    if lock.get("review") != review or lock.get("completeness_evidence") != evidence:
        raise ValueError("development lineage lock disagrees with its human review source")
    return {"path": str(lock_path.resolve()), "sha256": sha256_file(lock_path)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("lineage", type=Path)
    parser.add_argument("review", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(freeze_development_lineage(args.lineage, args.review, args.output), indent=2))


if __name__ == "__main__":
    main()
