"""Validate and materialize the governed V9 phone-review result.

The resulting lane is development diagnostic evidence. It is deliberately not an
automatic training dataset because all cases come from the three repeatedly used
development subjects and no verified occupant tracks are available.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

import cv2

from training.common import sha256_file
from training.data_paths import data_path
from training.dmd.v9_diagnostics import DEVELOPMENT_SUBJECTS, PROTECTED_HOLDOUTS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v9_research"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _canonical_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode()
    return hashlib.sha256(encoded).hexdigest()


def _indexed_proposals(row: dict[str, Any]) -> dict[str, dict[str, Any]]:
    result = {}
    for sample in row.get("samples", []):
        for proposal in sample.get("proposals", []):
            proposal_id = str(proposal["proposal_id"])
            if proposal_id in result:
                raise ValueError(f"duplicate proposal ID in {row['case_id']}: {proposal_id}")
            result[proposal_id] = {**proposal, "sample": sample}
    return result


def validate_reviews(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    source_path = output / "phone_proposal_mining.json"
    diagnostics_path = output / "v9_diagnostics.json"
    reviews_dir = output / "human_phone_reviews"
    source = _read(source_path)
    diagnostics = _read(diagnostics_path)
    rows = {row["case_id"]: row for row in source["rows"]}
    diagnostic_rows = {
        row["case_id"]: row for row in diagnostics["phone_false_negatives"]["cases"]
    }
    receipt_paths = sorted(reviews_dir.glob("*.json"))
    receipts = {path.stem: _read(path) for path in receipt_paths}
    if len(rows) != 13 or set(rows) != set(receipts) or set(rows) != set(diagnostic_rows):
        raise ValueError("V9 human review must cover the exact frozen 13-case set")
    subjects = {str(row["subject_id"]) for row in rows.values()}
    if subjects != set(DEVELOPMENT_SUBJECTS) or subjects & set(PROTECTED_HOLDOUTS):
        raise ValueError("V9 human review contains an unexpected or protected subject")

    source_hash = sha256_file(source_path)
    validated = []
    for case_id in sorted(rows):
        row = rows[case_id]
        receipt = receipts[case_id]
        path = reviews_dir / f"{case_id}.json"
        revision = receipt.get("revision")
        if revision != _canonical_sha256(
            {key: value for key, value in receipt.items() if key != "revision"}
        ):
            raise ValueError(f"receipt revision is invalid: {case_id}")
        if receipt.get("case_id") != case_id or receipt.get("reviewer_type") != "HUMAN":
            raise ValueError(f"receipt lacks human provenance: {case_id}")
        if receipt.get("production_approval") is not False:
            raise ValueError(f"receipt makes an invalid production claim: {case_id}")
        if not str(receipt.get("training_admission", "")).startswith("NOT_AUTOMATIC"):
            raise ValueError(f"receipt bypasses governed materialization: {case_id}")
        source_info = receipt.get("source") or {}
        if source_info.get("proposal_artifact_sha256") != source_hash:
            raise ValueError(f"receipt references a different proposal artifact: {case_id}")
        if source_info.get("source_case_sha256") != _canonical_sha256(row):
            raise ValueError(f"receipt source-case hash is invalid: {case_id}")
        contact_sheet = ROOT / row["contact_sheet"]
        if source_info.get("contact_sheet_sha256") != sha256_file(contact_sheet):
            raise ValueError(f"receipt contact-sheet hash is invalid: {case_id}")
        if receipt.get("phone_visibility") != "PHONE_VISIBLE":
            raise ValueError(f"review did not confirm phone visibility: {case_id}")
        if receipt.get("phone_use_assessment") != "PHONE_USE_CONFIRMED":
            raise ValueError(f"review did not confirm phone use: {case_id}")

        proposal_id = str(receipt.get("selected_proposal_id"))
        proposals = _indexed_proposals(row)
        proposal = None if proposal_id == "NO_VALID_PROPOSAL" else proposals.get(proposal_id)
        if proposal_id != "NO_VALID_PROPOSAL" and proposal is None:
            raise ValueError(f"receipt selects an unknown proposal: {case_id}")
        expected = (
            None
            if proposal is None
            else {
                "proposal_id": proposal["proposal_id"],
                "frame_index": int(proposal["sample"]["frame_index"]),
                "timestamp_seconds": float(proposal["sample"]["timestamp_seconds"]),
                "source": proposal["source"],
                "confidence": float(proposal["confidence"]),
                "geometry_context": proposal["geometry_context"],
                "hand_proximity": float(proposal["hand_proximity"]),
                "face_proximity": float(proposal["face_proximity"]),
                "nearest_cross_model_iou": float(proposal["nearest_cross_model_iou"]),
                "xyxy_in_driver_roi": [
                    float(value) for value in proposal["xyxy_in_driver_roi"]
                ],
            }
        )
        if receipt.get("selected_proposal") != expected:
            raise ValueError(f"receipt proposal snapshot is stale: {case_id}")
        validated.append(
            {
                "case_id": case_id,
                "subject_id": row["subject_id"],
                "action": diagnostic_rows[case_id]["action"],
                "reviewer_id": receipt["reviewer_id"],
                "reviewed_at_utc": receipt["reviewed_at_utc"],
                "receipt_path": path.relative_to(ROOT).as_posix(),
                "receipt_sha256": sha256_file(path),
                "receipt_revision": revision,
                "phone_visibility": receipt["phone_visibility"],
                "phone_use_assessment": receipt["phone_use_assessment"],
                "selected_proposal_id": proposal_id,
                "selected_proposal": expected,
                "selected_sample": (
                    None
                    if proposal is None
                    else {
                        "frame_index": int(proposal["sample"]["frame_index"]),
                        "timestamp_seconds": float(proposal["sample"]["timestamp_seconds"]),
                        "driver_roi_xyxy_in_full_frame": [
                            int(value)
                            for value in proposal["sample"]["driver_roi_xyxy_in_full_frame"]
                        ],
                    }
                ),
                "notes": receipt["notes"],
                "roi_source": diagnostic_rows[case_id]["roi_source"],
                "occupant_track_status": "UNVERIFIED_NO_OCCUPANT_TRACK",
            }
        )

    valid = [item for item in validated if item["selected_proposal"] is not None]
    hard = [item for item in validated if item["selected_proposal"] is None]
    confidences = [float(item["selected_proposal"]["confidence"]) for item in valid]
    snapshot = _canonical_sha256(
        [{"case_id": item["case_id"], "revision": item["receipt_revision"]} for item in validated]
    )
    return {
        "schema_version": 1,
        "completed_at_utc": max(item["reviewed_at_utc"] for item in validated),
        "scope": "HUMAN_REVIEWED_DEVELOPMENT_DIAGNOSTIC_ONLY",
        "review_snapshot_sha256": snapshot,
        "source_proposal_artifact": source_path.relative_to(ROOT).as_posix(),
        "source_proposal_artifact_sha256": source_hash,
        "cases": validated,
        "summary": {
            "cases": len(validated),
            "human_reviewed": len(validated),
            "phone_visible": sum(item["phone_visibility"] == "PHONE_VISIBLE" for item in validated),
            "phone_use_confirmed": sum(
                item["phone_use_assessment"] == "PHONE_USE_CONFIRMED" for item in validated
            ),
            "valid_phone_proposals": len(valid),
            "no_valid_proposal_hard_examples": len(hard),
            "selected_sources": dict(
                Counter(item["selected_proposal"]["source"] for item in valid)
            ),
            "selected_confidence_min": min(confidences),
            "selected_confidence_median": median(confidences),
            "selected_confidence_mean": mean(confidences),
            "selected_confidence_max": max(confidences),
            "verified_occupant_tracks": 0,
            "development_subjects": sorted(subjects),
        },
        "governance": {
            "training_admission": "DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_TRAINING_READY",
            "reason": (
                "All reviewed cases are FN-selected evidence from three repeatedly used development "
                "subjects; using them for training and reporting recovery on the same cases would leak."
            ),
            "hard_example_policy": (
                "PHONE_VISIBLE with NO_VALID_PROPOSAL is retained without an empty YOLO label and "
                "must never be treated as a phone-negative image."
            ),
            "protected_holdouts_accessed": False,
            "ground_truth_changed": False,
            "event_matcher_changed": False,
            "production_approved": False,
        },
    }


def _relative_or_absolute(path: Path) -> str:
    resolved = path.resolve()
    try:
        return resolved.relative_to(ROOT).as_posix()
    except ValueError:
        return str(resolved)


def materialize(result: dict[str, Any], output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output = output.resolve()
    snapshot = result["review_snapshot_sha256"]
    target = data_path("reviewed") / "v9_phone_diagnostic" / snapshot[:12]
    manifest_path = output / "human_phone_review_materialization.json"
    if target.exists():
        if not manifest_path.is_file():
            raise FileExistsError(f"materialized lane exists without a manifest: {target}")
        existing = _read(manifest_path)
        if existing.get("review_snapshot_sha256") != snapshot:
            raise FileExistsError("materialized lane belongs to a different review snapshot")
        for item in existing["items"]:
            for path_key, hash_key in (("image_path", "image_sha256"), ("label_path", "label_sha256")):
                value = item.get(path_key)
                if value and sha256_file(ROOT / value) != item[hash_key]:
                    raise ValueError(f"materialized artifact hash mismatch: {value}")
        return existing

    positives_dir = target / "positives"
    hard_dir = target / "hard_examples"
    positives_dir.mkdir(parents=True)
    hard_dir.mkdir(parents=True)
    external_root = data_path("external_dmd")
    items = []
    for item in result["cases"]:
        case_id = item["case_id"]
        proposal = item["selected_proposal"]
        if proposal is None:
            source = ROOT / item["roi_source"]
            image_target = hard_dir / f"{case_id}{source.suffix.lower()}"
            shutil.copy2(source, image_target)
            items.append(
                {
                    "case_id": case_id,
                    "subject_id": item["subject_id"],
                    "kind": "VISIBLE_PHONE_NO_VALID_BOX_HARD_EXAMPLE",
                    "image_path": _relative_or_absolute(image_target),
                    "image_sha256": sha256_file(image_target),
                    "label_path": None,
                    "label_sha256": None,
                    "training_admission": "EXCLUDED_NO_VALID_BOX",
                    "receipt_revision": item["receipt_revision"],
                }
            )
            continue

        sample = item["selected_sample"]
        videos = sorted((external_root / "subjects" / item["subject_id"]).glob("*_rgb_body.mp4"))
        if len(videos) != 1:
            raise ValueError(f"expected one body video for {item['subject_id']}, found {len(videos)}")
        video = videos[0]
        capture = cv2.VideoCapture(str(video))
        try:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(sample["frame_index"]))
            ok, frame = capture.read()
        finally:
            capture.release()
        if not ok:
            raise ValueError(f"could not decode selected frame for {case_id}")
        left, top, right, bottom = (int(value) for value in sample["driver_roi_xyxy_in_full_frame"])
        roi = frame[top:bottom, left:right]
        if roi.size == 0:
            raise ValueError(f"empty driver ROI for {case_id}")
        x1, y1, x2, y2 = (float(value) for value in proposal["xyxy_in_driver_roi"])
        height, width = roi.shape[:2]
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError(f"phone box escapes driver ROI for {case_id}")
        yolo = [
            ((x1 + x2) / 2) / width,
            ((y1 + y2) / 2) / height,
            (x2 - x1) / width,
            (y2 - y1) / height,
        ]
        stem = f"{case_id}__f{int(sample['frame_index'])}"
        image_target = positives_dir / f"{stem}.jpg"
        label_target = positives_dir / f"{stem}.txt"
        if not cv2.imwrite(str(image_target), roi, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise ValueError(f"could not write reviewed ROI for {case_id}")
        label_target.write_text("0 " + " ".join(f"{value:.8f}" for value in yolo) + "\n", encoding="utf-8")
        items.append(
            {
                "case_id": case_id,
                "subject_id": item["subject_id"],
                "kind": "HUMAN_CONFIRMED_MODEL_PROPOSAL",
                "frame_index": int(sample["frame_index"]),
                "source_group_id": f"DMD:{item['subject_id']}:{video.name}",
                "source_video_sha256": sha256_file(video),
                "selected_proposal_id": item["selected_proposal_id"],
                "proposal_source": proposal["source"],
                "proposal_confidence": float(proposal["confidence"]),
                "yolo": yolo,
                "image_path": _relative_or_absolute(image_target),
                "image_sha256": sha256_file(image_target),
                "label_path": _relative_or_absolute(label_target),
                "label_sha256": sha256_file(label_target),
                "training_admission": "DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_TRAINING_READY",
                "receipt_revision": item["receipt_revision"],
            }
        )
    manifest = {
        "schema_version": 1,
        "created_at_utc": datetime.now(UTC).isoformat(),
        "review_snapshot_sha256": snapshot,
        "output": _relative_or_absolute(target),
        "items": items,
        "summary": {
            "items": len(items),
            "positive_box_labels": sum(item["kind"] == "HUMAN_CONFIRMED_MODEL_PROPOSAL" for item in items),
            "visible_phone_hard_examples_without_labels": sum(
                item["kind"] == "VISIBLE_PHONE_NO_VALID_BOX_HARD_EXAMPLE" for item in items
            ),
            "automatic_training_items": 0,
        },
        "training_admission": "DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_TRAINING_READY",
        "production_approved": False,
    }
    _write(manifest_path, manifest)
    return manifest


def render_report(result: dict[str, Any], materialized: dict[str, Any], output: Path) -> Path:
    rows = []
    for item in result["cases"]:
        proposal = item["selected_proposal"]
        rows.append(
            "| "
            + " | ".join(
                [
                    item["case_id"],
                    item["action"].split("/")[-1],
                    item["selected_proposal_id"],
                    "—" if proposal is None else proposal["source"],
                    "—" if proposal is None else f"{100 * float(proposal['confidence']):.2f}%",
                    "VISIBLE_PHONE_NO_VALID_BOX" if proposal is None else "HUMAN_CONFIRMED_BOX",
                ]
            )
            + " |"
        )
    summary = result["summary"]
    report = f"""# V9 human phone review results

All 13 frozen development false negatives have an authenticated human receipt. Reviewers confirmed
phone visibility and phone use in 13/13 cases. Twelve cases have a selected proposal; one case
(`gB-9-phone-06`) has a visible phone but no valid proposal.

| Case | Action | Selected proposal | Source | Confidence | Materialization |
| --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

- Human-reviewed cases: {summary['human_reviewed']}/13.
- Valid selected phone proposals: {summary['valid_phone_proposals']}/13.
- Selected sources: `{summary['selected_sources']}`.
- Selected confidence range: {100 * summary['selected_confidence_min']:.2f}%–{100 * summary['selected_confidence_max']:.2f}% (median {100 * summary['selected_confidence_median']:.2f}%).
- Verified occupant tracks: 0/13.
- Materialized positive diagnostic labels: {materialized['summary']['positive_box_labels']}.
- Visible-phone hard examples without labels: {materialized['summary']['visible_phone_hard_examples_without_labels']}.
- Automatic training items: 0.

This result establishes that low-confidence auxiliary proposals can surface review candidates for these
FN-selected development cases. It does not estimate detector recall or justify a runtime threshold because
the cases were selected after failure and the chosen boxes came from the proposal pool. The materialized
lane is locked to `DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_TRAINING_READY`; training on it and measuring recovery
on the same three subjects would be leakage. `gB-9-phone-06` has no empty YOLO label, so its visible phone
cannot be learned as background. Protected holdouts remain sealed and production approval remains false.
"""
    path = output / "13_HUMAN_PHONE_REVIEW_RESULTS.md"
    path.write_text(report, encoding="utf-8")
    return path


def run(output: Path = DEFAULT_OUTPUT, *, materialize_lane: bool = True) -> dict[str, Any]:
    result = validate_reviews(output)
    _write(output / "human_phone_review_summary.json", result)
    materialized = materialize(result, output) if materialize_lane else {
        "summary": {
            "positive_box_labels": 0,
            "visible_phone_hard_examples_without_labels": 0,
            "automatic_training_items": 0,
        }
    }
    render_report(result, materialized, output)
    return {"review": result["summary"], "materialization": materialized["summary"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--no-materialize", action="store_true")
    args = parser.parse_args()
    print(json.dumps(run(args.output, materialize_lane=not args.no_materialize), indent=2))


if __name__ == "__main__":
    main()
