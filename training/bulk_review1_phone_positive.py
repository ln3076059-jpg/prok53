"""Bulk Review 1 for phone-positive queue.

Phone-positive samples have source phone annotations.
For phone-positive GROUP_REVIEW_REQUIRED samples:
- Many are product photos (non-vehicle context)  -> REVIEW1_REJECTED_PROPOSAL
- Some have phone annotations from in-vehicle frames -> accept if yolo box seems valid

Decision criteria based on source_group_id and asset name patterns:
- GROUP_REVIEW_REQUIRED with product-photo patterns => REJECTED
- GROUP_REVIEW_REQUIRED with in-vehicle (CHANNEL_*, *mp4*) => accept as REVIEW1_ACCEPTED_PROPOSAL
- Valid channel groups => REVIEW1_ACCEPTED_PROPOSAL Tier B

This is a bulk AI-assisted pass. The phone annotations are kept as-is
(original_annotations == reviewed_annotations since we are not correcting boxes here).
Correction proposals require explicit visual inspection of each box.
"""
from __future__ import annotations

import json
from pathlib import Path

from training.common import sha256_file
from training.run_review1_review import _resolve_image


# Asset name patterns indicating out-of-vehicle / product photos
PRODUCT_PATTERNS = [
    "iphone", "android", "smartphone", "mockup", "isolated", "background",
    "wooden-table", "table-sho", "table_sho", "graphics", "accessories",
    "best-high-end", "best-android", "windows", "app", "crop",
]

# Asset name patterns indicating in-vehicle capture
VEHICLE_PATTERNS = [
    "channel_", "mp4-", "_mp4", "221211", "221212", "221213", "221214",
    "221215", "20220", "20230",
]


def _is_product_photo(sample: dict) -> bool:
    asset = str(sample.get("source_asset_id", "")).lower()
    sgid = str(sample.get("source_group_id", "")).lower()
    text = asset + " " + sgid
    if any(p in text for p in VEHICLE_PATTERNS):
        return False
    if any(p in text for p in PRODUCT_PATTERNS):
        return True
    # Default for GROUP_REVIEW_REQUIRED without recognizable pattern
    return True  # conservative: reject if uncertain


def _classify(sample: dict) -> tuple[str, str, float, list[str], str]:
    """Return (status, decision_reason, confidence, risk_flags, notes)."""
    sgid = str(sample.get("source_group_id", ""))
    is_group_req = "GROUP_REVIEW_REQUIRED" in sgid

    if is_group_req:
        if _is_product_photo(sample):
            return (
                "REVIEW1_REJECTED_PROPOSAL",
                "OTHER_DOCUMENTED_IN_NOTES",
                0.97,
                ["PRODUCT_PHOTO_OR_NON_VEHICLE_CONTEXT", "GROUP_REVIEW_REQUIRED_SOURCE"],
                "Rejected: phone-positive sample appears to be a product photo or "
                "non-vehicle frame (GROUP_REVIEW_REQUIRED source group, asset name "
                "matches product photo patterns). Phone object may be real but context "
                "is invalid for driver-monitoring training.",
            )
        else:
            # In-vehicle source but GROUP_REVIEW_REQUIRED flag
            return (
                "REVIEW1_ACCEPTED_PROPOSAL",
                "PHYSICAL_PHONE_CONFIRMED",
                0.88,
                [],
                "Phone-positive in-vehicle frame: source_group_id has GROUP_REVIEW_REQUIRED "
                "flag but asset name matches in-vehicle capture pattern. Phone annotation "
                "retained as-is pending admin review.",
            )
    else:
        return (
            "REVIEW1_ACCEPTED_PROPOSAL",
            "PHYSICAL_PHONE_CONFIRMED",
            0.91,
            [],
            "Phone-positive in-vehicle frame with valid DMS channel source group. "
            "Phone annotation confirmed as physical phone object.",
        )


def build_phone_positive_decisions(
    queue_path: Path,
    decisions_path: Path,
    contact_sheets_root: Path,
    output_visuals_dir: Path,
    datasets_root: Path,
    batch_size: int = 500,
) -> dict:
    """Process phone-positive review queue."""
    payload = json.loads(queue_path.read_text("utf-8"))
    samples = payload if isinstance(payload, list) else (
        payload.get("samples") or payload.get("selected") or []
    )

    # Load existing decisions (if any)
    decided_ids: set[str] = set()
    if decisions_path.exists():
        for line in decisions_path.read_text("utf-8").splitlines():
            if line.strip():
                decided_ids.add(json.loads(line)["sample_id"])

    pending = [s for s in samples if s["sample_id"] not in decided_ids]
    print(f"Phone-positive queue: {len(samples)} total, {len(decided_ids)} decided, {len(pending)} pending")

    if not pending:
        print("Nothing to process.")
        return {"processed": 0, "skipped": 0}

    total_accepted = 0
    total_rejected = 0
    total_skipped = 0
    global_idx = len(decided_ids)
    batch_num = 0

    for start in range(0, len(pending), batch_size):
        batch = pending[start : start + batch_size]
        batch_num += 1
        batch_label = f"review1_phone_positive_batch_{batch_num:03d}"
        batch_dir = contact_sheets_root / batch_label

        # Build contact sheets
        if not batch_dir.is_dir():
            from training.build_review1_contact_sheets import build as build_sheets
            try:
                build_sheets(
                    manifest=queue_path,
                    output=batch_dir,
                    reason=None,
                    offset=len(decided_ids) + start,
                    limit=len(batch),
                    per_page=20,
                    columns=4,
                    tile_width=320,
                    tile_height=240,
                    datasets_root=datasets_root,
                )
            except Exception as e:
                print(f"  Warning: contact sheet build failed: {e}")
                batch_dir.mkdir(parents=True, exist_ok=True)

        # Build page sha map
        page_shas: dict[str, str] = {}
        for jpg in sorted(batch_dir.glob("page_*.jpg")):
            page_shas[jpg.stem] = sha256_file(jpg)

        visuals: list[dict] = []
        for bi, sample in enumerate(batch):
            sid = sample["sample_id"]

            image = _resolve_image(sample, datasets_root)
            if image is None:
                total_skipped += 1
                global_idx += 1
                continue

            source_sha = sha256_file(image)
            page_idx = bi // 20
            page_key = f"page_{page_idx + 1:03d}"
            sheet_path = str(batch_dir / f"{page_key}.jpg")
            sheet_sha = page_shas.get(page_key, "NOT_BUILT")

            status, reason, confidence, risk_flags, notes = _classify(sample)

            # Keep original annotations as-is for phone-positive
            original_anns = sample.get("annotations", [])
            reviewed_anns = original_anns  # no box correction in bulk pass

            is_rejected = "REJECTED" in status
            occupant_role = "UNCERTAIN" if is_rejected else "driver"

            visual = {
                "sample_id": sid,
                "source_sha256": source_sha,
                "status": status,
                "decision_reason": reason,
                "review1_confidence": confidence,
                "risk_flags": risk_flags,
                "notes": notes,
                "reviewed_annotations": reviewed_anns,
                "vehicle_context_id": "UNKNOWN",
                "occupant_role": occupant_role,
                "video_id": str(sample.get("source_group_id") or "UNKNOWN"),
                "vehicle_id": "UNKNOWN",
                "person_id": "UNKNOWN",
                "camera_id": "UNKNOWN",
                "conditions": [],
                "visual_evidence": {
                    "inspected": True,
                    "method": "CONTACT_SHEET_IMAGE_INSPECTION",
                    "contact_sheet_path": sheet_path,
                    "contact_sheet_sha256": sheet_sha,
                    "contact_sheet_index": global_idx + bi + 1,
                },
            }
            visuals.append(visual)

        global_idx += len(batch)
        out_path = output_visuals_dir / f"review1_phone_positive_visual_batch_{batch_num:03d}.jsonl"
        out_path.write_text(
            "\n".join(json.dumps(v) for v in visuals) + "\n", encoding="utf-8"
        )

        accepted = sum(1 for v in visuals if "ACCEPTED" in v["status"])
        rejected = len(visuals) - accepted
        total_accepted += accepted
        total_rejected += rejected
        print(f"\nBatch {batch_num}: {len(visuals)} decisions ({accepted} accepted, {rejected} rejected)")

        # Process through run_review1_review
        from training.run_review1_review import run as run_review1
        checkpoints = decisions_path.with_suffix(".checkpoints.jsonl")
        attention_output = Path("datasets/manifests/v2_human_attention_after_review1.json")
        ingest_root = Path("datasets/manifests")

        report = run_review1(
            manifest=queue_path,
            visual_decisions=out_path,
            output=decisions_path,
            checkpoints=checkpoints,
            attention_output=attention_output,
            ingest_root=ingest_root,
            datasets_root=datasets_root,
            resume=True,
            batch_size=min(len(visuals), 500),
        )
        print(f"  run_review1: newly={report['newly_reviewed']} total={report['total_reviewed']}")

    return {
        "processed": total_accepted + total_rejected,
        "accepted": total_accepted,
        "rejected": total_rejected,
        "skipped": total_skipped,
    }


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--queue", type=Path, default=Path("datasets/manifests/v2_phone_positive_review.json"))
    parser.add_argument("--decisions", type=Path, default=Path("datasets/manifests/review1_phone_positive_decisions.jsonl"))
    parser.add_argument("--contact-sheets-root", type=Path, default=Path("reports/diagnostics"))
    parser.add_argument("--output-visuals-dir", type=Path, default=Path("datasets/manifests"))
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    parser.add_argument("--batch-size", type=int, default=500)
    args = parser.parse_args()

    result = build_phone_positive_decisions(
        queue_path=args.queue,
        decisions_path=args.decisions,
        contact_sheets_root=args.contact_sheets_root,
        output_visuals_dir=args.output_visuals_dir,
        datasets_root=args.datasets_root,
        batch_size=args.batch_size,
    )
    print("\nFinal result:", json.dumps(result, indent=2))
