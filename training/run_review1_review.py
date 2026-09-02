from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

from training.common import YoloLabel, sha256_file, stable_json_hash

REVIEWER_ID = "review1"
REVIEWER_TYPE = "AI"
REVIEW_POLICY_VERSION = "review1_visual_policy_v1"
REVIEW1_STATUSES = {
    "REVIEW1_APPROVED_UNDER_ADMIN_DELEGATION",
    "REVIEW1_ACCEPTED_PROPOSAL",
    "REVIEW1_REJECTED_PROPOSAL",
    "REVIEW1_CORRECTION_PROPOSAL",
    "REVIEW1_UNCERTAIN",
    "NEEDS_HUMAN_REVIEW",
}
HUMAN_ONLY_STATUSES = {"APPROVED", "APPROVED_NEGATIVE"}
VISUAL_METHODS = {"DIRECT_IMAGE_INSPECTION", "CONTACT_SHEET_IMAGE_INSPECTION"}


def _read_jsonl(path: Path) -> list[dict]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _load_queue(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    for key in ("samples", "selected"):
        if isinstance(payload.get(key), list):
            return payload[key]
    raise ValueError("queue must be a list or contain a samples/selected list")


def _ingest_hashes(root: Path) -> dict[str, str]:
    hashes: dict[str, str] = {}
    for path in sorted(root.glob("ingest_*.jsonl")):
        for item in _read_jsonl(path):
            sample_id = str(item.get("sample_id", ""))
            digest = str(item.get("sha256", "")).lower()
            if not sample_id or len(digest) != 64:
                continue
            previous = hashes.setdefault(sample_id, digest)
            if previous != digest:
                raise ValueError(f"conflicting ingest hashes for sample {sample_id}")
    return hashes


def _resolve_image(item: dict, datasets_root: Path) -> Path:
    path = Path(str(item.get("image_path") or item.get("source_image") or ""))
    if not path.is_file():
        parts = path.parts
        dataset_index = next(
            (index for index, value in enumerate(parts) if value.lower() == "datasets"),
            None,
        )
        if dataset_index is not None:
            relocated = datasets_root.joinpath(*parts[dataset_index + 1 :])
            if relocated.is_file():
                path = relocated
    if not path.is_file():
        sample_id = str(item["sample_id"]).split(":box:", 1)[0]
        matches = [
            candidate
            for candidate in datasets_root.rglob(f"{sample_id}.*")
            if candidate.is_file()
        ]
        if len(matches) != 1:
            raise FileNotFoundError(
                f"expected one image for {item['sample_id']}, found {len(matches)}"
            )
        path = matches[0]
    resolved = path.resolve()
    root = datasets_root.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"unsafe image path outside datasets root: {resolved}")
    return resolved


def _annotations(values: list[dict], sample_id: str) -> list[dict]:
    result = []
    for index, value in enumerate(values):
        label = YoloLabel(
            int(value["class_id"]), *(float(component) for component in value["yolo"])
        )
        label.validate()
        result.append(
            {
                **value,
                "class_id": label.class_id,
                "yolo": [label.x_center, label.y_center, label.width, label.height],
                "box_id": value.get("box_id") or f"{sample_id}:box:{index}",
                "occupant_role": value.get("occupant_role", "UNCERTAIN"),
            }
        )
    return result


def _expected_hash(item: dict, ingest_hashes: dict[str, str]) -> str:
    sample_id = str(item["sample_id"]).split(":box:", 1)[0]
    value = str(item.get("sha256") or item.get("source_sha256") or "").lower()
    value = value if len(value) == 64 else ingest_hashes.get(sample_id, "")
    if len(value) != 64:
        raise ValueError(f"no immutable source SHA-256 for sample {item['sample_id']}")
    return value


def _validate_visual_decision(
    item: dict,
    visual: dict,
    actual_sha256: str,
    expected_sha256: str,
) -> dict:
    sample_id = str(item["sample_id"])
    if str(visual.get("sample_id")) != sample_id:
        raise ValueError(f"visual decision identity mismatch for {sample_id}")
    evidence = visual.get("visual_evidence") or {}
    if evidence.get("inspected") is not True or evidence.get("method") not in VISUAL_METHODS:
        raise ValueError(f"{sample_id} lacks explicit visual inspection evidence")
    if str(visual.get("source_sha256", "")).lower() != actual_sha256:
        raise ValueError(f"SOURCE_HASH_MISMATCH for visual decision {sample_id}")
    if actual_sha256 != expected_sha256:
        raise ValueError(f"SOURCE_HASH_MISMATCH for queue sample {sample_id}")

    status = str(visual.get("status", ""))
    if status in HUMAN_ONLY_STATUSES or status not in REVIEW1_STATUSES:
        raise ValueError(f"review1 cannot emit status {status!r}")
    confidence = float(visual.get("review1_confidence", -1))
    if not 0 <= confidence <= 1:
        raise ValueError(f"invalid review1_confidence for {sample_id}")
    risk_flags = sorted({str(value) for value in visual.get("risk_flags", []) if value})
    original = _annotations(item.get("annotations", []), sample_id)
    reviewed = _annotations(visual.get("reviewed_annotations", []), sample_id)
    reason = str(visual.get("decision_reason", "")).strip()
    if not reason:
        raise ValueError(f"missing decision_reason for {sample_id}")

    if status == "REVIEW1_APPROVED_UNDER_ADMIN_DELEGATION":
        if confidence < 0.97 or risk_flags:
            raise ValueError(f"Tier A requires confidence >= 0.97 and no risk flags: {sample_id}")
        tier = "TIER_A"
    elif status == "REVIEW1_ACCEPTED_PROPOSAL":
        if not 0.85 <= confidence < 0.97 or risk_flags:
            raise ValueError(f"Tier B requires 0.85 <= confidence < 0.97 and no risks: {sample_id}")
        tier = "TIER_B"
    else:
        tier = "TIER_C"

    if reason == "PHONE_MISSING_LABEL":
        original_phone = sum(value["class_id"] == 0 for value in original)
        reviewed_phone = sum(value["class_id"] == 0 for value in reviewed)
        if status != "REVIEW1_CORRECTION_PROPOSAL" or reviewed_phone <= original_phone:
            raise ValueError("PHONE_MISSING_LABEL requires an added phone correction proposal")
    if reason == "UNCERTAIN_OR_OCCLUDED" and any(
        value["class_id"] == 2 for value in reviewed
    ):
        raise ValueError("uncertain belt evidence cannot create an unfastened annotation")

    return {
        "schema_version": 3,
        "sample_id": sample_id,
        "previous_status": str(
            item.get("review_status") or item.get("human_review_status") or "PENDING"
        ),
        "new_status": status,
        "status": status,
        "tier": tier,
        "reviewer_id": REVIEWER_ID,
        "reviewer_type": REVIEWER_TYPE,
        "delegated_by": "admin",
        "approval_authority_id": "admin",
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
        "decision_reason": reason,
        "notes": str(visual.get("notes", "")),
        "source_sha256": actual_sha256,
        "source_group_id": item.get("source_group_id"),
        "source_id": item.get("source_id"),
        "source_asset_id": item.get("source_asset_id"),
        "original_annotations": original,
        "reviewed_annotations": reviewed,
        "annotations": reviewed,
        "review1_confidence": confidence,
        "risk_flags": risk_flags,
        "visual_evidence": evidence,
        "vehicle_context_id": visual.get("vehicle_context_id") or "UNKNOWN",
        "occupant_role": visual.get("occupant_role") or "UNCERTAIN",
        "video_id": visual.get("video_id") or item.get("video_id") or "UNKNOWN",
        "vehicle_id": visual.get("vehicle_id") or "UNKNOWN",
        "person_id": visual.get("person_id") or "UNKNOWN",
        "camera_id": visual.get("camera_id") or "UNKNOWN",
        "conditions": sorted({str(value) for value in visual.get("conditions", [])}),
        "needs_admin_confirmation": status in {
            "REVIEW1_APPROVED_UNDER_ADMIN_DELEGATION",
            "REVIEW1_ACCEPTED_PROPOSAL",
        },
        "human_confirmation": False,
        "governance_eligible": False,
        "review_policy_version": REVIEW_POLICY_VERSION,
    }


def _write_attention_queue(
    records: list[dict],
    output: Path,
    audit_fraction: float,
    seed: int,
    unreviewed: list[dict] | None = None,
) -> list[dict]:
    mandatory = [
        item
        for item in records
        if item["tier"] == "TIER_C"
        or item["new_status"] == "REVIEW1_CORRECTION_PROPOSAL"
        or item["risk_flags"]
    ]
    tier_a = sorted(
        (item for item in records if item["tier"] == "TIER_A"),
        key=lambda value: value["sample_id"],
    )
    randomizer = random.Random(seed)
    audit_count = round(len(tier_a) * audit_fraction)
    audit_ids = {
        item["sample_id"] for item in randomizer.sample(tier_a, min(audit_count, len(tier_a)))
    }
    selected = {item["sample_id"]: item for item in mandatory}
    for item in tier_a:
        if item["sample_id"] in audit_ids:
            selected[item["sample_id"]] = {**item, "attention_reason": "DETERMINISTIC_TIER_A_AUDIT"}
    for item in unreviewed or []:
        selected[str(item["sample_id"])] = {
            **item,
            "attention_reason": "NOT_REVIEWED_BY_REVIEW1",
            "new_status": "NEEDS_HUMAN_REVIEW",
            "reviewer_id": None,
            "reviewer_type": None,
            "human_confirmation": False,
            "governance_eligible": False,
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        try:
            existing = json.loads(output.read_text(encoding="utf-8"))
            existing_by_id = {str(item["sample_id"]): item for item in existing}
            existing_by_id.update(selected)
            queue = [existing_by_id[key] for key in sorted(existing_by_id)]
        except json.JSONDecodeError:
            queue = [selected[key] for key in sorted(selected)]
    else:
        queue = [selected[key] for key in sorted(selected)]
    output.write_text(json.dumps(queue, indent=2, sort_keys=True), encoding="utf-8")
    return queue


def run(
    manifest: Path,
    visual_decisions: Path,
    output: Path,
    checkpoints: Path,
    attention_output: Path,
    ingest_root: Path,
    datasets_root: Path,
    *,
    resume: bool = False,
    limit: int | None = None,
    samples: set[str] | None = None,
    min_confidence: float = 0.0,
    batch_size: int = 100,
    audit_fraction: float = 0.05,
    seed: int = 42,
    dry_run: bool = False,
) -> dict:
    if not 100 <= batch_size <= 500:
        raise ValueError("batch_size must be between 100 and 500")
    if not 0 <= audit_fraction <= 0.1:
        raise ValueError("audit_fraction must be between 0 and 0.1")
    if output.exists() and not resume:
        raise FileExistsError(f"output exists; use --resume: {output}")

    queue = _load_queue(manifest)
    queue_by_id = {str(item["sample_id"]): item for item in queue}
    if len(queue_by_id) != len(queue):
        raise ValueError("queue contains duplicate sample_id values")
    visuals = _read_jsonl(visual_decisions)
    visual_by_id = {str(item.get("sample_id")): item for item in visuals}
    if len(visual_by_id) != len(visuals):
        raise ValueError("visual decision input contains duplicate sample_id values")
    unknown = sorted(set(visual_by_id) - set(queue_by_id))
    if unknown:
        raise ValueError(f"visual decisions not present in queue: {unknown[:5]}")

    existing = _read_jsonl(output)
    existing_by_id = {str(item["sample_id"]): item for item in existing}
    candidates = [item for item in queue if str(item["sample_id"]) in visual_by_id]
    if samples:
        candidates = [item for item in candidates if str(item["sample_id"]) in samples]
    candidates = [
        item
        for item in candidates
        if float(visual_by_id[str(item["sample_id"])].get("review1_confidence", -1))
        >= min_confidence
        and str(item["sample_id"]) not in existing_by_id
    ]
    if limit is not None:
        candidates = candidates[:limit]

    hashes = _ingest_hashes(ingest_root)
    prepared = []
    for item in candidates:
        image = _resolve_image(item, datasets_root)
        actual = sha256_file(image).lower()
        expected = _expected_hash(item, hashes)
        record = _validate_visual_decision(
            item, visual_by_id[str(item["sample_id"])], actual, expected
        )
        record["source"] = {
            "queue_path": str(manifest),
            "queue_sha256": sha256_file(manifest),
            "image_path": str(image),
        }
        record["decision_id"] = stable_json_hash(record)
        prepared.append(record)

    if not dry_run:
        output.parent.mkdir(parents=True, exist_ok=True)
        checkpoints.parent.mkdir(parents=True, exist_ok=True)
        for start in range(0, len(prepared), batch_size):
            batch = prepared[start : start + batch_size]
            with output.open("a", encoding="utf-8") as handle:
                for record in batch:
                    handle.write(json.dumps(record, sort_keys=True) + "\n")
            checkpoint = {
                "schema_version": 1,
                "batch_id": stable_json_hash([item["decision_id"] for item in batch]),
                "start_index": len(existing) + start,
                "end_index": len(existing) + start + len(batch) - 1,
                "reviewed_count": len(batch),
                "tier_a": sum(item["tier"] == "TIER_A" for item in batch),
                "tier_b": sum(item["tier"] == "TIER_B" for item in batch),
                "tier_c": sum(item["tier"] == "TIER_C" for item in batch),
                "corrections": sum(
                    item["new_status"] == "REVIEW1_CORRECTION_PROPOSAL" for item in batch
                ),
                "rejects": sum(
                    item["new_status"] == "REVIEW1_REJECTED_PROPOSAL" for item in batch
                ),
                "uncertain": sum(
                    item["new_status"] in {"REVIEW1_UNCERTAIN", "NEEDS_HUMAN_REVIEW"}
                    for item in batch
                ),
                "batch_sha256": stable_json_hash(batch),
                "created_at_utc": datetime.now(UTC).isoformat(),
            }
            with checkpoints.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(checkpoint, sort_keys=True) + "\n")

    all_records = existing + ([] if dry_run else prepared)
    reviewed_ids = {str(record["sample_id"]) for record in all_records}
    attention = (
        []
        if dry_run
        else _write_attention_queue(
            all_records,
            attention_output,
            audit_fraction,
            seed,
            [item for item in queue if str(item["sample_id"]) not in reviewed_ids],
        )
    )
    tiers = Counter(item["tier"] for item in all_records)
    statuses = Counter(item["new_status"] for item in all_records)
    return {
        "status": "DRY_RUN" if dry_run else "REVIEW1_BATCH_COMPLETE",
        "reviewer_id": REVIEWER_ID,
        "reviewer_type": REVIEWER_TYPE,
        "delegated_by": "admin",
        "queue": str(manifest),
        "queue_sha256": sha256_file(manifest),
        "total_queue": len(queue),
        "total_reviewed": len(all_records),
        "newly_reviewed": len(prepared),
        "tier_counts": dict(tiers),
        "status_counts": dict(statuses),
        "human_attention_remaining": len(attention),
        "human_confirmed": sum(
            item.get("reviewer_type") == "HUMAN"
            and item.get("new_status") in HUMAN_ONLY_STATUSES
            for item in all_records
        ),
        "governed_ready": False,
        "output": str(output),
        "checkpoints": str(checkpoints),
        "attention_queue": str(attention_output),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Run append-only Review 1 visual decisions")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--visual-decisions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path)
    parser.add_argument(
        "--attention-output",
        type=Path,
        default=Path("datasets/manifests/v2_human_attention_after_review1.json"),
    )
    parser.add_argument("--ingest-root", type=Path, default=Path("datasets/manifests"))
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--sample", action="append", default=[])
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--batch-size", type=int, default=100)
    parser.add_argument("--audit-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    checkpoints = args.checkpoints or args.output.with_suffix(".checkpoints.jsonl")
    report = run(
        args.manifest,
        args.visual_decisions,
        args.output,
        checkpoints,
        args.attention_output,
        args.ingest_root,
        args.datasets_root,
        resume=args.resume,
        limit=args.limit,
        samples=set(args.sample),
        min_confidence=args.min_confidence,
        batch_size=args.batch_size,
        audit_fraction=args.audit_fraction,
        seed=args.seed,
        dry_run=args.dry_run,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
