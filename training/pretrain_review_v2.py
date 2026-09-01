from __future__ import annotations

import argparse
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

import cv2

from training.common import YoloLabel, sha256_file
from training.propose_phone_boxes import iou

SCHEMA_VERSION = 1
REVIEWER_MODEL = "roadwatch_pretrain_reviewer_v1"
VALID_TASKS = {"phone", "seatbelt", "uncertain", "mixed"}
MIN_UNCERTAIN_COUNTS = {"train": 100, "val": 25, "test": 25}


def load_queue(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(payload, list):
        return payload
    selected = payload.get("selected")
    if isinstance(selected, list):
        return selected
    raise ValueError(f"queue must be a list or contain selected: {path}")


def resolve_image(item: dict, incoming_root: Path = Path("datasets/incoming")) -> Path:
    path = Path(str(item.get("image_path", "")))
    if path.is_file():
        return path
    sample_id = str(item["sample_id"])
    matches = [
        candidate for candidate in incoming_root.rglob(f"{sample_id}.*") if candidate.is_file()
    ]
    if len(matches) != 1:
        raise FileNotFoundError(
            f"expected one image for {sample_id} after stale-path recovery, found {len(matches)}"
        )
    return matches[0]


def image_evidence(path: Path) -> dict:
    image = cv2.imread(str(path))
    if image is None:
        raise ValueError(f"cannot decode image: {path}")
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(gray.mean())
    contrast = float(gray.std())
    blur = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    height, width = gray.shape
    suggested_conditions = []
    if brightness < 55:
        suggested_conditions.append("low_light")
    elif brightness >= 75:
        suggested_conditions.append("daylight_or_well_lit")
    if blur < 55:
        suggested_conditions.append("motion_blur_or_defocus")
    if contrast < 25:
        suggested_conditions.append("low_contrast")
    return {
        "width": width,
        "height": height,
        "brightness": round(brightness, 3),
        "contrast": round(contrast, 3),
        "laplacian_variance": round(blur, 3),
        "suggested_conditions": suggested_conditions,
        "condition_provenance": "MODEL_HEURISTIC_REQUIRES_CONFIRMATION",
    }


def _annotation_evidence(item: dict, width: int, height: int) -> dict:
    valid = []
    errors = []
    short_sides = []
    for index, annotation in enumerate(item.get("annotations", [])):
        try:
            label = YoloLabel(
                int(annotation["class_id"]),
                *(float(value) for value in annotation["yolo"]),
            )
            label.validate()
        except (KeyError, TypeError, ValueError) as exc:
            errors.append(f"box {index}: {exc}")
            continue
        valid.append(annotation)
        short_sides.append(round(min(label.width * width, label.height * height), 3))
    return {
        "count": len(item.get("annotations", [])),
        "valid_count": len(valid),
        "errors": errors,
        "short_side_pixels": short_sides,
        "classes": sorted({int(annotation["class_id"]) for annotation in valid}),
    }


def _normalized_xyxy(yolo: Iterable[float]) -> list[float]:
    x, y, width, height = (float(value) for value in yolo)
    return [x - width / 2, y - height / 2, x + width / 2, y + height / 2]


def _phone_agreement(item: dict, detections: list[dict]) -> dict:
    proposed = [
        annotation
        for annotation in item.get("annotations", [])
        if int(annotation.get("class_id", -1)) == 0
    ]
    matches = []
    for annotation in proposed:
        proposal_box = _normalized_xyxy(annotation["yolo"])
        best = max(
            (
                {
                    "iou": round(iou(proposal_box, detection["xyxy_normalized"]), 6),
                    "confidence": detection["confidence"],
                }
                for detection in detections
            ),
            key=lambda value: value["iou"],
            default={"iou": 0.0, "confidence": 0.0},
        )
        matches.append(best)
    return {
        "detected_phones": len(detections),
        "max_detection_confidence": round(
            max((value["confidence"] for value in detections), default=0.0), 6
        ),
        "proposal_matches": matches,
        "independent_model_agreement": bool(matches)
        and all(value["iou"] >= 0.2 and value["confidence"] >= 0.15 for value in matches),
    }


def review_item(
    item: dict,
    task: str,
    quality: dict,
    phone_detections: list[dict] | None = None,
) -> dict:
    if task not in VALID_TASKS:
        raise ValueError(f"unsupported task: {task}")
    annotations = _annotation_evidence(item, quality["width"], quality["height"])
    phone = _phone_agreement(item, phone_detections or [])
    reasons = []
    severe_quality = quality["brightness"] < 35 or quality["laplacian_variance"] < 25
    marginal_quality = (
        quality["brightness"] < 55 or quality["contrast"] < 25 or quality["laplacian_variance"] < 55
    )
    tiny = any(value < 8 for value in annotations["short_side_pixels"])
    source_errors = list(item.get("source_label_errors", []))

    decision = "PASS"
    tier = "MEDIUM_CONFIDENCE"
    confidence = 0.72
    evidence_level = "TECHNICAL_AND_SOURCE_CONSENSUS"

    if annotations["errors"] or source_errors:
        decision = "UNCERTAIN"
        tier = "UNCERTAIN"
        confidence = 0.2
        reasons.append("invalid_or_clipped_source_annotation")
    elif severe_quality or tiny:
        decision = "UNCERTAIN"
        tier = "UNCERTAIN"
        confidence = 0.35
        if task == "uncertain":
            evidence_level = "INSUFFICIENT_VISIBILITY_HEURISTIC"
        reasons.append("severe_visibility_or_tiny_object_risk")
    elif task == "uncertain":
        decision = "UNCERTAIN"
        tier = "UNCERTAIN"
        confidence = 0.9
        evidence_level = "PURPOSE_BUILT_UNCERTAINTY_CANDIDATE"
        reasons.append("candidate_selected_for_occlusion_or_visibility_review")
    elif task in {"phone", "mixed"} and annotations["classes"] == [0]:
        if phone["independent_model_agreement"]:
            tier = "HIGH_CONFIDENCE"
            confidence = min(0.99, 0.82 + 0.15 * phone["max_detection_confidence"])
            evidence_level = "SOURCE_AND_INDEPENDENT_PHONE_MODEL_AGREEMENT"
            reasons.append("phone_box_agrees_with_coco_phone_detector")
        else:
            confidence = 0.68 if not marginal_quality else 0.58
            reasons.append("source_phone_box_lacks_independent_model_agreement")
    elif task in {"phone", "mixed"} and not item.get("annotations"):
        if phone["detected_phones"]:
            decision = "UNCERTAIN"
            tier = "UNCERTAIN"
            confidence = max(0.5, phone["max_detection_confidence"])
            evidence_level = "POSSIBLE_MISSING_PHONE_LABEL"
            reasons.append("phone_detector_found_candidate_in_proposed_negative")
        else:
            confidence = 0.7 if not marginal_quality else 0.58
            evidence_level = "SOURCE_NEGATIVE_PLUS_MODEL_NON_DETECTION"
            reasons.append("no_source_phone_box_and_no_model_phone_detection")
    elif task in {"seatbelt", "mixed"} and set(annotations["classes"]).issubset({1, 2}):
        if marginal_quality:
            confidence = 0.62
            reasons.append("seatbelt_state_source_label_with_marginal_visibility")
        else:
            tier = "HIGH_CONFIDENCE"
            confidence = 0.88
            reasons.append("seatbelt_state_source_label_and_valid_comparable_roi")
        evidence_level = "SOURCE_STATE_AND_ROI_QUALITY_NOT_INDEPENDENT_VISION"
    else:
        decision = "UNCERTAIN"
        tier = "UNCERTAIN"
        confidence = 0.25
        reasons.append("task_and_annotation_schema_do_not_agree")

    training_eligible = decision == "PASS" or (
        task == "uncertain"
        and decision == "UNCERTAIN"
        and evidence_level == "INSUFFICIENT_VISIBILITY_HEURISTIC"
    )
    return {
        "schema_version": SCHEMA_VERSION,
        "status": "MODEL_ASSISTED_PROPOSAL_PENDING_APPROVAL",
        "reviewer_type": "AUTOMATED",
        "reviewer_model": REVIEWER_MODEL,
        "reviewed_at_utc": datetime.now(UTC).isoformat(),
        "decision": decision,
        "tier": tier,
        "confidence": round(float(confidence), 6),
        "reason": reasons,
        "evidence_level": evidence_level,
        "checks": {
            "image_quality": quality,
            "annotations": annotations,
            "phone_model": phone,
            "source_group_present": bool(item.get("source_group_id")),
            "source_label_errors": source_errors,
        },
        "metadata_suggestions": {
            "video_id": item.get("source_group_id"),
            "conditions": quality["suggested_conditions"],
            "provenance": "MODEL_SUGGESTED_REQUIRES_CONFIRMATION",
        },
        "training_eligible": training_eligible,
        "training_target": (
            "uncertain_or_occluded" if task == "uncertain" else "canonical_annotations"
        ),
        "governance_eligible": False,
        "requires_human_confirmation": tier != "HIGH_CONFIDENCE",
        "policy": (
            "Model-assisted review may gate exploratory training but never creates HUMAN_APPROVED."
        ),
    }


def _phone_predictions(
    items: list[dict], model_path: Path | None, device: str, batch: int
) -> tuple[dict[str, list[dict]], dict]:
    if model_path is None:
        return {}, {"enabled": False}
    from ultralytics import YOLO

    model = YOLO(str(model_path))
    phone_ids = {
        int(class_id)
        for class_id, name in model.names.items()
        if str(name).lower().replace("_", " ") in {"phone", "cell phone", "mobile phone"}
    }
    if not phone_ids:
        raise ValueError(f"model has no phone class: {model.names}")
    unique = {item["sample_id"]: resolve_image(item) for item in items}
    predictions: dict[str, list[dict]] = {}
    sample_ids = list(unique)
    for start in range(0, len(sample_ids), batch):
        chunk_ids = sample_ids[start : start + batch]
        results = model.predict(
            source=[str(unique[sample_id]) for sample_id in chunk_ids],
            imgsz=640,
            conf=0.05,
            device=device,
            verbose=False,
        )
        for sample_id, result in zip(chunk_ids, results, strict=True):
            values = []
            for detected in result.boxes:
                if int(detected.cls.item()) not in phone_ids:
                    continue
                values.append(
                    {
                        "confidence": round(float(detected.conf.item()), 6),
                        "xyxy_normalized": [float(value) for value in detected.xyxyn[0].tolist()],
                    }
                )
            predictions[sample_id] = values
    return predictions, {
        "enabled": True,
        "path": str(model_path),
        "sha256": sha256_file(model_path),
        "phone_class_ids": sorted(phone_ids),
        "device": device,
    }


def run(
    queue_path: Path,
    output: Path,
    task: str,
    phone_model: Path | None = None,
    device: str = "cpu",
    batch: int = 16,
) -> dict:
    items = load_queue(queue_path)
    phone_predictions, model = _phone_predictions(
        items if task in {"phone", "mixed"} else [],
        phone_model if task in {"phone", "mixed"} else None,
        device,
        batch,
    )
    reviewed = []
    errors = []
    for item in items:
        try:
            path = resolve_image(item)
            review = review_item(
                item,
                task,
                image_evidence(path),
                phone_predictions.get(item["sample_id"], []),
            )
            reviewed.append({**item, "image_path": str(path.resolve()), "proposal_review": review})
        except (FileNotFoundError, ValueError) as exc:
            errors.append({"sample_id": item.get("sample_id"), "error": str(exc)})
    counts = Counter(row["proposal_review"]["tier"] for row in reviewed)
    decisions = Counter(row["proposal_review"]["decision"] for row in reviewed)
    report = {
        "schema_version": SCHEMA_VERSION,
        "status": "PASS" if not errors and len(reviewed) == len(items) else "FAIL",
        "queue": str(queue_path),
        "queue_sha256": sha256_file(queue_path),
        "queue_samples": len(items),
        "reviewed_samples": len(reviewed),
        "reviewer_model": REVIEWER_MODEL,
        "phone_model": model,
        "tier_counts": dict(sorted(counts.items())),
        "decision_counts": dict(sorted(decisions.items())),
        "training_eligible": sum(row["proposal_review"]["training_eligible"] for row in reviewed),
        "human_confirmation_required": sum(
            row["proposal_review"]["requires_human_confirmation"] for row in reviewed
        ),
        "errors": errors,
        "policy": (
            "MODEL_ASSISTED_PENDING_APPROVAL is sufficient only for explicitly opted-in "
            "exploratory training. It is not HUMAN_APPROVED or governed ground truth."
        ),
        "samples": reviewed,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run auditable model-assisted V2 pretrain review")
    parser.add_argument("queue", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--task", choices=sorted(VALID_TASKS), required=True)
    parser.add_argument("--phone-model", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    args = parser.parse_args()
    report = run(
        args.queue,
        args.output,
        args.task,
        args.phone_model,
        args.device,
        args.batch,
    )
    print(json.dumps({key: value for key, value in report.items() if key != "samples"}, indent=2))
    raise SystemExit(0 if report["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
