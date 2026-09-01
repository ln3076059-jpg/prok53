from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import cv2

from training.propose_phone_boxes import iou

PHONE_FOLDERS = {
    "01_phone_negative_model_conflict",
    "02_phone_negative_low_visibility",
    "03_phone_positive_low_visibility",
    "06_external_adt_uncertain",
    "07_external_mendeley_uncertain",
    "08_hard_examples_uncertain",
}
HUMAN_DRAFT_FOLDER = "00_user_reviewed_pending_identity"
SEATBELT_UNCERTAIN_FOLDER = "05_seatbelt_uncertain_classifier_candidates"


def _read_jsonl(path: Path) -> list[dict]:
    return [
        json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()
    ]


def _source_annotations() -> dict[str, list[dict]]:
    result = {}
    for path in Path("datasets/manifests/pretrain_pending_approval").glob("*.json"):
        payload = json.loads(path.read_text(encoding="utf-8"))
        for item in payload.get("samples", []):
            result[str(item["sample_id"])] = item.get("annotations", [])
    return result


def _detections(result) -> list[dict]:
    values = []
    for box in result.boxes:
        values.append(
            {
                "class_id": int(box.cls),
                "confidence": round(float(box.conf), 6),
                "xyxy_normalized": [round(float(value), 8) for value in box.xyxyn[0]],
            }
        )
    return values


def _second_pass_decision(folder: str, annotations: list[dict], detections: list[dict]) -> dict:
    phones = [item for item in detections if item["class_id"] == 67]
    people = [item for item in detections if item["class_id"] == 0]
    vehicles = [item for item in detections if item["class_id"] in {2, 5, 7}]
    strong_phone = max((item["confidence"] for item in phones), default=0.0)
    person_confidence = max((item["confidence"] for item in people), default=0.0)
    vehicle_confidence = max((item["confidence"] for item in vehicles), default=0.0)
    proposal_boxes = [
        annotation for annotation in annotations if int(annotation.get("class_id", -1)) == 0
    ]
    agreement = []
    for proposal in proposal_boxes:
        x, y, width, height = (float(value) for value in proposal["yolo"])
        normalized = [x - width / 2, y - height / 2, x + width / 2, y + height / 2]
        agreement.append(
            max(
                (iou(normalized, item["xyxy_normalized"]) for item in phones),
                default=0.0,
            )
        )

    if folder == "03_phone_positive_low_visibility":
        if strong_phone >= 0.25 and agreement and max(agreement) >= 0.2:
            decision = "PHONE_POSITIVE_SECOND_MODEL_AGREEMENT"
        elif strong_phone >= 0.25:
            decision = "PHONE_DETECTED_BUT_SOURCE_BOX_DISAGREES"
        else:
            decision = "REMAINS_UNCERTAIN_SOURCE_PHONE_ONLY"
    elif strong_phone >= 0.25:
        decision = "PHONE_PRESENT_PROPOSAL"
    elif strong_phone >= 0.08:
        decision = "PHONE_POSSIBLE_REVIEW_REQUIRED"
    else:
        decision = "NO_PHONE_DETECTED_SECOND_PASS"
    return {
        "decision": decision,
        "max_phone_confidence": round(strong_phone, 6),
        "max_person_confidence": round(person_confidence, 6),
        "max_vehicle_confidence": round(vehicle_confidence, 6),
        "source_phone_box_max_iou": round(max(agreement, default=0.0), 6),
        "detections": detections,
    }


def _write_records(path: Path, records: dict[str, dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = [records[key] for key in sorted(records)]
    path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in ordered) + "\n",
        encoding="utf-8",
    )


def review(root: Path, output: Path, model_path: Path, batch: int) -> dict:
    existing_rows = _read_jsonl(output) if output.is_file() else []
    existing = {item["worklist_key"]: item for item in existing_rows}
    annotations = _source_annotations()
    pending_model = []
    records = dict(existing)
    integrity_failures = []

    for folder in sorted(path for path in root.iterdir() if path.is_dir()):
        for item in _read_jsonl(folder / "manifest.jsonl"):
            key = f"{folder.name}/{item['sample_id']}"
            if key in records:
                continue
            image_path = folder / item["image"]
            image = cv2.imread(str(image_path))
            if image is None:
                integrity_failures.append(key)
                continue
            base = {
                "worklist_key": key,
                "sample_id": item["sample_id"],
                "folder": folder.name,
                "image_path": str(image_path.resolve()),
                "source_proposal_review": {
                    "decision": item.get("proposal_decision"),
                    "tier": item.get("proposal_tier"),
                    "confidence": item.get("proposal_confidence"),
                    "reasons": item.get("reasons", []),
                },
                "reviewer_type": "AUTOMATED",
                "reviewer_model": "yolo11s_coco_pretrain_review",
                "approval_status": "PENDING_HUMAN_APPROVAL",
                "governance_eligible": False,
            }
            if folder.name == HUMAN_DRAFT_FOLDER:
                records[key] = {**base, "decision": "HUMAN_DRAFT_TAKES_PRECEDENCE"}
            elif folder.name == SEATBELT_UNCERTAIN_FOLDER:
                records[key] = {**base, "decision": "KEEP_UNCERTAIN_OR_OCCLUDED"}
            elif folder.name not in PHONE_FOLDERS:
                records[key] = {**base, "decision": "SEATBELT_VISUAL_REVIEW_STILL_REQUIRED"}
            else:
                pending_model.append((key, image_path, base))

    if pending_model:
        from ultralytics import YOLO

        model = YOLO(str(model_path))
        _write_records(output, records)
        chunk_size = max(batch * 2, 1)
        completed = 0
        for start in range(0, len(pending_model), chunk_size):
            chunk = pending_model[start : start + chunk_size]
            results = model.predict(
                [str(item[1]) for item in chunk],
                imgsz=640,
                conf=0.05,
                classes=[0, 2, 5, 7, 67],
                device="cpu",
                batch=batch,
                verbose=False,
            )
            for pending, result in zip(chunk, results, strict=True):
                key, _, base = pending
                evidence = _second_pass_decision(
                    base["folder"],
                    annotations.get(base["sample_id"], []),
                    _detections(result),
                )
                records[key] = {**base, **evidence}
            completed += len(chunk)
            _write_records(output, records)
            print(
                f"reviewed {completed}/{len(pending_model)} model-assisted images",
                flush=True,
            )

    ordered = [records[key] for key in sorted(records)]
    _write_records(output, records)
    counts = Counter(item["decision"] for item in ordered)
    summary = {
        "schema_version": 1,
        "status": "INTEGRITY_PASS" if not integrity_failures else "INTEGRITY_FAIL",
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
        "reviewer_type": "AUTOMATED",
        "reviewer_model": "yolo11s_coco_pretrain_review",
        "approval_status": "PENDING_HUMAN_APPROVAL",
        "model_path": str(model_path),
        "images_checked": len(ordered),
        "decision_counts": dict(counts),
        "integrity_failures": integrity_failures,
        "governance_eligible": False,
        "policy": (
            "This is a model-assisted pretrain review pending human approval. Seatbelt "
            "semantic uncertainty is retained when the general COCO model cannot verify "
            "belt state."
        ),
    }
    summary_path = output.with_suffix(".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run a model-assisted pretrain review pending human approval"
    )
    parser.add_argument(
        "--root", type=Path, default=Path("datasets/review_worklists/v2_pending_approval")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v2_review_pending_approval.jsonl"),
    )
    parser.add_argument("--model", type=Path, default=Path("models/base/yolo11s.pt"))
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    print(json.dumps(review(args.root, args.output, args.model, args.batch), indent=2))


if __name__ == "__main__":
    main()
