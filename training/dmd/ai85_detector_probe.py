"""Fixed-resolution phone proposals on original FN ROIs, for diagnosis only.

This probe does not label phones, associate them with drivers, or evaluate events.
Its paired resolution comparison tests whether pixels lost by the old 64px action
cache remain usable by the frozen phone detector.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from training.dmd.ai85_audit import artifact, read_json, write_json


def run(output: Path) -> dict:
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    model_path = Path("models/active/v2/phone_detector.pt")
    model = YOLO(str(model_path))
    rows = []
    for item in read_json(output / "fn_frame_manifest.json"):
        if item["status"] != "EXPORTED":
            continue
        image_path = Path(item["path"])
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"cannot decode diagnostic ROI: {image_path}")
        for resolution in (384, 960):
            started = time.perf_counter()
            prediction = model.predict(
                image, imgsz=resolution, conf=0.10, iou=0.7,
                device="cpu", verbose=False, max_det=20,
            )[0]
            proposals = [{
                "xyxy_in_roi": box.xyxy[0].cpu().tolist(),
                "confidence": float(box.conf.item()),
                "class_id": int(box.cls.item()),
                "class_name": prediction.names[int(box.cls.item())],
            } for box in prediction.boxes]
            rows.append({
                "case_id": item["case_id"], "timestamp_seconds": item["timestamp_seconds"],
                "input_roi": artifact(image_path), "imgsz": resolution,
                "seconds": time.perf_counter() - started, "proposals": proposals,
                "phone_candidate_present": any(p["class_name"] == "phone" for p in proposals),
                "review_status": "UNVERIFIED_OBJECT_PROPOSALS_NOT_VIOLATIONS",
            })
        print(f"phone ROI probe completed: {item['case_id']}", flush=True)
    result = {
        "model": artifact(model_path), "resolutions": [384, 960], "confidence": 0.10,
        "rows": rows,
        "proposal_coverage": {str(size): {
            "cases": sum(r["imgsz"] == size for r in rows),
            "cases_with_phone_proposals": sum(r["imgsz"] == size and
                                               r["phone_candidate_present"] for r in rows),
        } for size in (384, 960)},
        "limitations": [
            "FN-selected diagnostic snapshots, not an independent detection evaluation.",
            "No phone boxes or driver-use labels were inferred from these proposals.",
            "Proposal presence does not establish phone correctness or event recovery.",
        ],
    }
    write_json(output / "phone_detector_probe.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/ai85_upgrade"))
    args = parser.parse_args()
    result = run(args.output)
    print(result["proposal_coverage"])


if __name__ == "__main__":
    main()
