"""Frozen-weight spatial probes and seatbelt inference on consumed development data."""

from __future__ import annotations

import argparse
import csv
import gc
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import ultralytics
from ultralytics import YOLO

from training.dmd.adaptive_sampling import padded_roi, restore_boxes
from training.dmd.ai85_audit import artifact, read_json, write_json
from training.dmd.ai85_next import DEVELOPMENT_SUBJECTS
from training.dmd.holdout_guard import assert_not_consumed_holdout_for_development
from training.dmd.run_ai85_plus_temporal import timing
from training.dmd.run_v5_experiments import discover_external_dmd_root
from training.evaluate_seatbelt_v6 import CLASSES, evaluate_rows

OUT = Path("reports/ai85_plus")


def inventory() -> None:
    rows = read_json(OUT / "audit.json")["model_inventory"]
    by_hash = {}
    for row in rows:
        path = Path(row["path"])
        if row["sha256"] in by_hash:
            row.update(by_hash[row["sha256"]])
            continue
        metadata = {}
        try:
            if path.suffix == ".npz":
                with np.load(path) as data:
                    metadata = {
                        "architecture": "balanced_logistic_regression",
                        "class_mapping": data["classes"].tolist(),
                        "training_dataset": data["training_subjects"].tolist(),
                        "version": str(data["task"]),
                        "input_resolution": "512-dimensional frozen embedding",
                    }
            elif path.suffix in {".pt", ".pth"}:
                checkpoint = torch.load(path, map_location="cpu", weights_only=False)
                if isinstance(checkpoint, dict):
                    model = checkpoint.get("ema") or checkpoint.get("model")
                    args = checkpoint.get("train_args", {}) or {}
                    metadata = {
                        "architecture": type(model).__name__ if model is not None else "state_dict",
                        "class_mapping": getattr(model, "names", None),
                        "training_dataset": args.get("data", checkpoint.get("training_subjects")),
                        "input_resolution": args.get("imgsz"),
                        "epochs": args.get("epochs"),
                        "checkpoint_epoch": checkpoint.get("epoch"),
                        "version": checkpoint.get("version", checkpoint.get("variant")),
                        "validation_results": checkpoint.get("train_metrics"),
                        "has_optimizer_state": checkpoint.get("optimizer") is not None,
                        "has_scheduler_state": checkpoint.get("scheduler") is not None,
                        "resume_status": "EXACT_RESUME_NOT_VERIFIED",
                        "training_subjects": checkpoint.get("training_subjects"),
                        "heldout_subject": checkpoint.get("heldout_subject"),
                    }
                    if "model_state_dict" in checkpoint and "hierarchical" in path.name:
                        metadata["architecture"] = "FrozenResNet18_HierarchicalBiGRU"
                        metadata["class_mapping"] = [
                            "SAFE_DRIVING",
                            "TEXTING_LEFT",
                            "TEXTING_RIGHT",
                            "PHONECALL_LEFT",
                            "PHONECALL_RIGHT",
                            "OTHER_ACTIVITY",
                        ]
                        metadata["input_resolution"] = "64->112; sequence 9 x 512"
                    del checkpoint, model
            metadata["approval_status"] = "RESEARCH_ONLY_NO_NEW_PRODUCTION_APPROVAL"
        except Exception as exc:
            metadata["inspection_error"] = f"{type(exc).__name__}: {exc}"
        by_hash[row["sha256"]] = metadata
        row.update(metadata)
        gc.collect()
    write_json(
        OUT / "checkpoint_metadata.json",
        {
            "models": rows,
            "unknown_fields": "null means absent in checkpoint; not inferred from filename",
            "note": "Existing active deployment is preserved; this audit grants no approval.",
        },
    )
    print(f"inventoried {len(rows)} checkpoints", flush=True)


def phone() -> None:
    model_path = Path("models/active/v2/phone_detector.pt")
    model = YOLO(str(model_path))
    if list(model.names.values()) != ["phone"]:
        raise ValueError("phone checkpoint class map incompatible")
    cases = read_json("reports/ai85_next/fn_recovery.json")["cases"]
    root = discover_external_dmd_root()
    configs = [
        ("A0_full_640", 640, False, False, "full", 0),
        ("A1_full_640_TTA", 640, True, False, "full", 0),
        ("A2_full_960", 960, False, False, "full", 0),
        ("A2_full_1024", 1024, False, False, "full", 0),
        ("A3_full_640_agnostic", 640, False, True, "full", 0),
        ("A4_driver_640", 640, False, False, "driver", 0),
        ("A4_padded_driver_640", 640, False, False, "driver", 0.15),
    ]
    output = []
    for subject in DEVELOPMENT_SUBJECTS:
        assert_not_consumed_holdout_for_development(subject, "ai85_plus_phone_probe")
        video = next((root / "subjects" / subject).glob("*_rgb_body.mp4"))
        cap = cv2.VideoCapture(str(video))
        fps = cap.get(cv2.CAP_PROP_FPS)
        try:
            for case in (c for c in cases if c["subject_id"] == subject):
                start, end = case["truth"]["start_seconds"], case["truth"]["end_seconds"]
                target = round((start + end) * 0.5 * fps)
                cap.set(cv2.CAP_PROP_POS_FRAMES, target)
                ok, frame = cap.read()
                if not ok:
                    raise ValueError(f"decode failed {subject} {target}")
                row = {
                    "case_id": case["case_id"],
                    "subject": subject,
                    "frame_index": target,
                    "timestamp_seconds": target / fps,
                    "inside_truth_interval": start <= target / fps <= end,
                    "visual_review_inherited": case["visual_review"],
                    "review_scope": "one midpoint, not entire event",
                    "variants": [],
                }
                for name, size, tta, agnostic, mode, padding in configs:
                    roi = (
                        (0, 0, frame.shape[1], frame.shape[0])
                        if mode == "full"
                        else padded_roi(frame.shape, (0, 0.1, 0.65, 0.95), padding)
                    )
                    crop = frame[roi[1] : roi[3], roi[0] : roi[2]]
                    t0 = time.perf_counter()
                    result = model.predict(
                        crop,
                        imgsz=size,
                        conf=0.1,
                        iou=0.7,
                        augment=tta,
                        agnostic_nms=agnostic,
                        device="cpu",
                        verbose=False,
                        max_det=20,
                    )[0]
                    elapsed = time.perf_counter() - t0
                    boxes = restore_boxes(result.boxes.xyxy.cpu().numpy(), roi, frame.shape)
                    proposals = [
                        {
                            "original_xyxy": b.tolist(),
                            "confidence": float(conf),
                            "width_pixels": float(b[2] - b[0]),
                            "height_pixels": float(b[3] - b[1]),
                            "association": "UNVERIFIED_STATIC_ROI_NOT_DRIVER_TRACK",
                        }
                        for b, conf in zip(boxes, result.boxes.conf.cpu().numpy(), strict=True)
                    ]
                    row["variants"].append(
                        {
                            "id": name,
                            "imgsz": size,
                            "tta": tta,
                            "agnostic_nms": agnostic,
                            "roi": roi,
                            "seconds": elapsed,
                            "proposals": proposals,
                        }
                    )
                output.append(row)
                print(f"phone spatial probes {case['case_id']}", flush=True)
        finally:
            cap.release()
    summary = []
    for name, *_ in configs:
        entries = [next(v for v in row["variants"] if v["id"] == name) for row in output]
        summary.append(
            {
                "id": name,
                "frames": len(entries),
                "frames_with_proposal": sum(bool(v["proposals"]) for v in entries),
                "proposals": sum(len(v["proposals"]) for v in entries),
                "latency": timing([v["seconds"] for v in entries]),
            }
        )
    write_json(
        OUT / "phone_spatial_probes.json",
        {
            "model": artifact(model_path),
            "ultralytics": ultralytics.__version__,
            "confidence": 0.1,
            "nms_iou": 0.7,
            "scope": "FN-selected midpoint diagnostic only; no bbox ground truth",
            "AP50": None,
            "AP50_95": None,
            "event_recovery_measured": False,
            "class_agnostic_note": "Single-class phone checkpoint: no cross-class NMS interaction.",
            "A5": "NOT_SELECTED_NO_LEGITIMATE_VALIDATION_GAIN",
            "rows": output,
            "summary": summary,
        },
    )


def seatbelt() -> None:
    source = Path("reports/calibration/seatbelt_classifier_raw_predictions.csv")
    with source.open(encoding="utf-8", newline="") as stream:
        original = list(csv.DictReader(stream))
    root = Path("dataset/derived/v2_pretrain_pending_approval/seatbelt_classifier/val")
    images = [root / r["true_class"] / r["image"] for r in original]
    missing = [p.as_posix() for p in images if not p.is_file()]
    if missing:
        raise FileNotFoundError(f"validation crops missing: {missing[:3]}")
    model_path = Path("models/active/v2/seatbelt_classifier.pt")
    model = YOLO(str(model_path))
    if tuple(model.names.values()) != tuple(CLASSES):
        raise ValueError("seatbelt class mapping incompatible")
    rows = []
    for name, size, flip in (
        ("S0_original_224", 224, False),
        ("S1_resolution_320", 320, False),
        ("S2_horizontal_flip_TTA_224", 224, True),
    ):
        predictions, times = [], []
        for i, (item, path) in enumerate(zip(original, images, strict=True)):
            frame = cv2.imread(str(path))
            if frame is None:
                raise ValueError(f"cannot decode {path}")
            t0 = time.perf_counter()
            p = (
                model.predict(frame, imgsz=size, verbose=False, device="cpu")[0]
                .probs.data.cpu()
                .numpy()
            )
            if flip:
                flipped = (
                    model.predict(cv2.flip(frame, 1), imgsz=size, verbose=False, device="cpu")[0]
                    .probs.data.cpu()
                    .numpy()
                )
                p = (p + flipped) / 2
            times.append(time.perf_counter() - t0)
            predictions.append(
                {
                    "image": item["image"],
                    "true_class": item["true_class"],
                    "prob_fastened": str(float(p[0])),
                    "prob_unfastened": str(float(p[1])),
                    "prob_uncertain": str(float(p[2])),
                }
            )
            if (i + 1) % 160 == 0:
                print(f"seatbelt {name}: {i + 1}/{len(images)}", flush=True)
        path = OUT / f"{name}_predictions.csv"
        with path.open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(predictions[0]))
            writer.writeheader()
            writer.writerows(predictions)
        rows.append(
            {
                "id": name,
                "imgsz": size,
                "horizontal_flip_average": flip,
                "evaluation": evaluate_rows(predictions),
                "latency": timing(times),
                "predictions": artifact(path),
                "selection_eligible": False,
            }
        )
    write_json(
        OUT / "seatbelt_results.json",
        {
            "model": artifact(model_path),
            "ultralytics": ultralytics.__version__,
            "original_csv": artifact(source),
            "original_csv_replay": evaluate_rows(original),
            "variants": rows,
            "input_manifest": [artifact(p) for p in images],
            "scope": "641 consumed validation crops; MODEL_ASSISTED_PENDING_APPROVAL labels",
            "independent": False,
            "fine_tuning": "BLOCKED_LABEL_APPROVAL_AND_INDEPENDENCE",
        },
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=["inventory", "phone", "seatbelt"])
    parser.add_argument("--output", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.output
    OUT.mkdir(parents=True, exist_ok=True)
    filename = {
        "inventory": "checkpoint_metadata.json",
        "phone": "phone_spatial_probes.json",
        "seatbelt": "seatbelt_results.json",
    }[args.task]
    if (OUT / filename).exists():
        raise FileExistsError("use --output with a fresh directory to preserve experiment history")
    torch.set_num_threads(1)
    cv2.setNumThreads(1)
    {"inventory": inventory, "phone": phone, "seatbelt": seatbelt}[args.task]()
