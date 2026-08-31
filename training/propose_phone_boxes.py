from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from training.ingest_dataset import read_source_labels


def yolo_to_xyxy(box: list[float], width: int, height: int) -> tuple[float, float, float, float]:
    x_center, y_center, box_width, box_height = box
    return (
        (x_center - box_width / 2) * width,
        (y_center - box_height / 2) * height,
        (x_center + box_width / 2) * width,
        (y_center + box_height / 2) * height,
    )


def expanded_crop(
    box: tuple[float, float, float, float], width: int, height: int, padding: float
) -> tuple[int, int, int, int]:
    left, top, right, bottom = box
    pad_x = (right - left) * padding
    pad_y = (bottom - top) * padding
    return (
        max(0, int(left - pad_x)),
        max(0, int(top - pad_y)),
        min(width, int(np.ceil(right + pad_x))),
        min(height, int(np.ceil(bottom + pad_y))),
    )


def crop_xyxy_to_full_yolo(
    crop_box: tuple[int, int, int, int],
    detected_xyxy: tuple[float, float, float, float],
    image_width: int,
    image_height: int,
) -> list[float]:
    crop_left, crop_top, _, _ = crop_box
    left, top, right, bottom = detected_xyxy
    left += crop_left
    right += crop_left
    top += crop_top
    bottom += crop_top
    return [
        ((left + right) / 2) / image_width,
        ((top + bottom) / 2) / image_height,
        (right - left) / image_width,
        (bottom - top) / image_height,
    ]


def iou(left: list[float], right: list[float]) -> float:
    lx1, ly1, lx2, ly2 = left
    rx1, ry1, rx2, ry2 = right
    intersection = max(0.0, min(lx2, rx2) - max(lx1, rx1)) * max(
        0.0, min(ly2, ry2) - max(ly1, ry1)
    )
    union = (lx2 - lx1) * (ly2 - ly1) + (rx2 - rx1) * (ry2 - ry1) - intersection
    return 0.0 if union <= 0 else intersection / union


def deduplicate(proposals: list[dict], threshold: float = 0.5) -> list[dict]:
    kept: list[dict] = []
    for proposal in sorted(proposals, key=lambda item: item["confidence"], reverse=True):
        if all(iou(proposal["xyxy_pixels"], item["xyxy_pixels"]) < threshold for item in kept):
            kept.append(proposal)
    return kept


def load_records(path: Path, source_id: str) -> list[dict]:
    return [
        item
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
        for item in [json.loads(line)]
        if item.get("source_id") == source_id and item.get("status") != "EXACT_DUPLICATE"
    ]


def propose(
    records: list[dict],
    source_class_id: int,
    model_name: str,
    confidence: float,
    image_size: int,
    padding: float,
    chunk_size: int,
    device: str,
) -> dict:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise RuntimeError("install requirements.txt before generating proposals") from exc

    model = YOLO(model_name)
    phone_class_ids = {
        int(class_id)
        for class_id, name in model.names.items()
        if str(name).strip().lower().replace("_", " ") in {"cell phone", "mobile phone", "phone"}
    }
    if not phone_class_ids:
        raise ValueError(f"model {model_name!r} has no phone class: {model.names}")

    tasks: list[dict] = []
    samples: dict[str, dict] = {}
    for record in records:
        label_path = Path(record["source_label_path"]) if record.get("source_label_path") else None
        if label_path is None:
            continue
        behavior_boxes = [box for class_id, box in read_source_labels(label_path) if class_id == source_class_id]
        if not behavior_boxes:
            continue
        image_path = Path(record["original_path"])
        with Image.open(image_path) as image:
            rgb = image.convert("RGB")
            width, height = rgb.size
            samples[record["sample_id"]] = {
                "sample_id": record["sample_id"],
                "image_path": record["ingested_path"],
                "source_asset_id": record["source_asset_id"],
                "source_group_id": record["source_group_id"],
                "vehicle_context_required": True,
                "occupant_role": "PENDING",
                "source_behavior_class_id": source_class_id,
                "proposals": [],
            }
            for behavior_box in behavior_boxes:
                crop_box = expanded_crop(yolo_to_xyxy(behavior_box, width, height), width, height, padding)
                tasks.append(
                    {
                        "sample_id": record["sample_id"],
                        "image": np.asarray(rgb.crop(crop_box)),
                        "crop_box": crop_box,
                        "image_width": width,
                        "image_height": height,
                        "source_behavior_yolo": behavior_box,
                    }
                )

    for start in range(0, len(tasks), chunk_size):
        chunk = tasks[start : start + chunk_size]
        results = model.predict(
            source=[task["image"] for task in chunk],
            conf=confidence,
            imgsz=image_size,
            device=device,
            verbose=False,
        )
        for task, result in zip(chunk, results, strict=True):
            for detected in result.boxes:
                class_id = int(detected.cls.item())
                if class_id not in phone_class_ids:
                    continue
                crop_xyxy = tuple(float(value) for value in detected.xyxy[0].tolist())
                full_yolo = crop_xyxy_to_full_yolo(
                    task["crop_box"], crop_xyxy, task["image_width"], task["image_height"]
                )
                x_center, y_center, box_width, box_height = full_yolo
                xyxy_pixels = [
                    (x_center - box_width / 2) * task["image_width"],
                    (y_center - box_height / 2) * task["image_height"],
                    (x_center + box_width / 2) * task["image_width"],
                    (y_center + box_height / 2) * task["image_height"],
                ]
                samples[task["sample_id"]]["proposals"].append(
                    {
                        "class_id": 0,
                        "class_name": "phone",
                        "confidence": round(float(detected.conf.item()), 6),
                        "yolo": [round(value, 8) for value in full_yolo],
                        "xyxy_pixels": [round(value, 3) for value in xyxy_pixels],
                        "source_behavior_yolo": task["source_behavior_yolo"],
                        "requires_review": True,
                        "review_status": "PENDING",
                    }
                )

    output_samples = []
    for sample in samples.values():
        sample["proposals"] = deduplicate(sample["proposals"])
        sample["proposal_count"] = len(sample["proposals"])
        output_samples.append(sample)
    counts = Counter(sample["proposal_count"] for sample in output_samples)
    return {
        "schema_version": 1,
        "generator": {
            "model": model_name,
            "phone_class_ids": sorted(phone_class_ids),
            "confidence": confidence,
            "image_size": image_size,
            "crop_padding": padding,
            "policy": "Detector-assisted proposals only; human approval and occupant-role evidence remain mandatory.",
        },
        "summary": {
            "behavior_images": len(output_samples),
            "images_with_proposals": sum(bool(sample["proposal_count"]) for sample in output_samples),
            "total_proposals": sum(sample["proposal_count"] for sample in output_samples),
            "proposal_count_histogram": dict(sorted(counts.items())),
        },
        "samples": output_samples,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate review-only physical-phone proposals")
    parser.add_argument("ingest_manifest", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-class-id", type=int, default=3)
    parser.add_argument("--model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--padding", type=float, default=0.2)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = propose(
        load_records(args.ingest_manifest, args.source_id),
        args.source_class_id,
        args.model,
        args.confidence,
        args.image_size,
        args.padding,
        args.chunk_size,
        args.device,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report["summary"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
