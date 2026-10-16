"""Instrument frozen phone proposals at multiple timestamps for every AI85 false negative."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import cv2
import torch
from ultralytics import YOLO

from training.dmd.ai85_audit import artifact, read_json, write_json
from training.dmd.ai85_next import DEVELOPMENT_SUBJECTS, interval_sample_times
from training.dmd.run_v5_experiments import discover_external_dmd_root


def _proposal(box: Any, names: dict[int, str], width: int, height: int) -> dict[str, Any]:
    xyxy = [float(value) for value in box.xyxy[0].cpu().tolist()]
    center_x = (xyxy[0] + xyxy[2]) / (2.0 * width)
    center_y = (xyxy[1] + xyxy[3]) / (2.0 * height)
    class_id = int(box.cls.item())
    return {
        "xyxy_in_driver_roi": xyxy,
        "center_normalized_in_driver_roi": [center_x, center_y],
        "confidence": float(box.conf.item()),
        "class_id": class_id,
        "class_name": names[class_id],
        "occupant_association": "DRIVER_ROI_LEVEL_ONLY_NO_TRACK_ID",
    }


def run(output: Path, *, samples_per_event: int = 5, resolution: int = 960) -> dict[str, Any]:
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    root = discover_external_dmd_root()
    model_path = Path("models/active/v2/phone_detector.pt")
    model = YOLO(str(model_path))
    cases = read_json(output / "fn_recovery.json")["cases"]
    rows = []
    started = time.perf_counter()
    for subject in DEVELOPMENT_SUBJECTS:
        subject_cases = [row for row in cases if row["subject_id"] == subject]
        if not subject_cases:
            continue
        records = read_json(root / "derivatives" / f"v3_features_{subject}.json")["records"]
        video = next((root / "subjects" / subject).glob("*_rgb_body.mp4"))
        capture = cv2.VideoCapture(str(video))
        try:
            for case in subject_cases:
                truth = case["truth"]
                selected_records = []
                for target in interval_sample_times(
                    float(truth["start_seconds"]),
                    float(truth["end_seconds"]),
                    samples_per_event,
                ):
                    record = min(records, key=lambda row: abs(row["timestamp_seconds"] - target))
                    if record["frame_index"] not in {
                        existing["frame_index"] for existing in selected_records
                    }:
                        selected_records.append(record)
                samples = []
                for record in selected_records:
                    capture.set(cv2.CAP_PROP_POS_FRAMES, int(record["frame_index"]))
                    ok, frame = capture.read()
                    if not ok:
                        samples.append(
                            {
                                "frame_index": int(record["frame_index"]),
                                "timestamp_seconds": float(record["timestamp_seconds"]),
                                "decode_status": "FAILED",
                            }
                        )
                        continue
                    height, width = frame.shape[:2]
                    bounds = [0, int(0.10 * height), int(0.65 * width), int(0.95 * height)]
                    roi = frame[bounds[1] : bounds[3], bounds[0] : bounds[2]]
                    inference_started = time.perf_counter()
                    prediction = model.predict(
                        roi,
                        imgsz=resolution,
                        conf=0.10,
                        iou=0.7,
                        device="cpu",
                        verbose=False,
                        max_det=20,
                    )[0]
                    proposals = [
                        _proposal(box, prediction.names, roi.shape[1], roi.shape[0])
                        for box in prediction.boxes
                    ]
                    phone_proposals = [row for row in proposals if row["class_name"] == "phone"]
                    samples.append(
                        {
                            "frame_index": int(record["frame_index"]),
                            "timestamp_seconds": float(record["timestamp_seconds"]),
                            "decode_status": "OK",
                            "driver_roi_xyxy_in_full_frame": bounds,
                            "driver_roi_shape": list(roi.shape),
                            "inference_seconds": time.perf_counter() - inference_started,
                            "legacy_phone_confidences": {
                                view: float(record[f"{view}_phone_conf"])
                                for view in ("body", "face", "hands")
                            },
                            "proposals": proposals,
                            "phone_proposals": phone_proposals,
                        }
                    )
                rows.append(
                    {
                        "case_id": case["case_id"],
                        "truth": truth,
                        "requested_samples": samples_per_event,
                        "unique_samples": len(samples),
                        "samples": samples,
                        "samples_with_phone_proposal": sum(
                            bool(sample.get("phone_proposals")) for sample in samples
                        ),
                    }
                )
                print(f"detector trace completed: {case['case_id']}", flush=True)
        finally:
            capture.release()
    result = {
        "model": artifact(model_path),
        "resolution": resolution,
        "confidence_threshold": 0.10,
        "rows": rows,
        "cases": len(rows),
        "cases_with_any_phone_proposal": sum(
            row["samples_with_phone_proposal"] > 0 for row in rows
        ),
        "sampled_frames": sum(len(row["samples"]) for row in rows),
        "sampled_frames_with_phone_proposal": sum(
            row["samples_with_phone_proposal"] for row in rows
        ),
        "runtime_seconds": time.perf_counter() - started,
        "association_scope": "DRIVER_ROI_LEVEL_ONLY_NO_PHONE_OR_OCCUPANT_TRACK_ID",
        "limitations": [
            "FN-selected development intervals; not an independent detector evaluation.",
            "No ground-truth phone boxes exist, so proposal precision and recall are not measured.",
            "Driver ROI containment is not proof that the driver is using the detected object.",
        ],
    }
    write_json(output / "detector_trace.json", result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/ai85_next"))
    parser.add_argument("--samples-per-event", type=int, default=5)
    parser.add_argument("--resolution", type=int, default=960)
    args = parser.parse_args()
    result = run(
        args.output, samples_per_event=args.samples_per_event, resolution=args.resolution
    )
    print(
        {
            "cases": result["cases"],
            "cases_with_any_phone_proposal": result["cases_with_any_phone_proposal"],
            "sampled_frames": result["sampled_frames"],
        }
    )


if __name__ == "__main__":
    main()
