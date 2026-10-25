"""Frozen phone detector diagnostic on provisional bootstrap validation labels."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path


EXPECTED_CHECKPOINT_SHA256 = "840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54"
EXPECTED_VALIDATION_IMAGES = 1782
EXPECTED_PHONE_INSTANCES = 229
EXPECTED_PHONE_POSITIVE_IMAGES = 181
RESOLUTIONS = (640, 960, 1280)


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def mount(slug: str) -> Path:
    candidates = (
        Path("/kaggle/input") / slug,
        Path("/kaggle/input/datasets/longngoc11") / slug,
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def locate_dataset_root(mounted: Path) -> Path:
    for manifest in mounted.rglob("BOOTSTRAP_MANIFEST.json"):
        candidate = manifest.parent
        if (candidate / "images" / "val").is_dir() and (candidate / "labels" / "val").is_dir():
            return candidate
    raise FileNotFoundError("BOOTSTRAP_DATASET_ROOT_NOT_FOUND")


def build_phone_only_validation(source: Path, target: Path) -> dict:
    image_target = target / "images" / "val"
    label_target = target / "labels" / "val"
    image_target.mkdir(parents=True, exist_ok=True)
    label_target.mkdir(parents=True, exist_ok=True)
    images = sorted(
        path for path in (source / "images" / "val").iterdir()
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    positive_images = 0
    instances = 0
    for image in images:
        destination = image_target / image.name
        if not destination.exists():
            os.symlink(image, destination)
        source_label = source / "labels" / "val" / f"{image.stem}.txt"
        phone_rows = []
        if source_label.is_file():
            for row in source_label.read_text().splitlines():
                fields = row.split()
                if fields and int(float(fields[0])) == 0:
                    phone_rows.append(" ".join(fields))
        if phone_rows:
            positive_images += 1
            instances += len(phone_rows)
        (label_target / f"{image.stem}.txt").write_text(
            "\n".join(phone_rows) + ("\n" if phone_rows else "")
        )
    observed = {
        "images": len(images),
        "phone_positive_images": positive_images,
        "phone_instances": instances,
    }
    expected = {
        "images": EXPECTED_VALIDATION_IMAGES,
        "phone_positive_images": EXPECTED_PHONE_POSITIVE_IMAGES,
        "phone_instances": EXPECTED_PHONE_INSTANCES,
    }
    if observed != expected:
        raise ValueError(f"PHONE_VALIDATION_COUNTS_MISMATCH:{observed}")
    yaml_path = target / "phone_only.yaml"
    yaml_path.write_text(
        f"path: {target}\ntrain: images/val\nval: images/val\nnc: 1\nnames:\n  0: phone\n"
    )
    observed["yaml"] = str(yaml_path)
    return observed


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.187"],
        check=True,
        timeout=300,
    )
    import cv2
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("GPU_NOT_ALLOCATED")
    checkpoint_root = mount("roadwatch-ai85-plus-frozen-checkpoints-v8")
    checkpoint = checkpoint_root / "phone_detector.pt"
    if digest(checkpoint) != EXPECTED_CHECKPOINT_SHA256:
        raise ValueError("PHONE_CHECKPOINT_HASH_MISMATCH")
    bootstrap_mount = mount("roadwatch-mc-bootstrap-6500-v1")
    source_root = locate_dataset_root(bootstrap_mount)
    bootstrap_manifest = source_root / "BOOTSTRAP_MANIFEST.json"
    manifest = json.loads(bootstrap_manifest.read_text())
    if manifest.get("status") != "PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA":
        raise ValueError("UNEXPECTED_BOOTSTRAP_GOVERNANCE_STATUS")

    # Keep generated labels, symlinks, caches, and validation plots outside the
    # exported output directory. Only the compact audit JSON should be published.
    working_root = Path("/kaggle/temp/phone_component_validation")
    counts = build_phone_only_validation(source_root, working_root)
    model = YOLO(str(checkpoint))
    if list(model.names.values()) != ["phone"]:
        raise ValueError("PHONE_CHECKPOINT_CLASS_MAPPING_MISMATCH")

    variants = []
    for resolution in RESOLUTIONS:
        torch.cuda.reset_peak_memory_stats()
        started = time.perf_counter()
        metrics = model.val(
            data=counts["yaml"],
            split="val",
            imgsz=resolution,
            batch=32,
            conf=0.001,
            iou=0.70,
            device=0,
            workers=4,
            plots=False,
            save_json=False,
            project="/kaggle/temp/validation_runs",
            name=f"phone_{resolution}",
            exist_ok=True,
            verbose=False,
        )
        elapsed = time.perf_counter() - started
        variants.append(
            {
                "resolution": resolution,
                "precision": float(metrics.box.mp),
                "recall": float(metrics.box.mr),
                "map50": float(metrics.box.map50),
                "map50_95": float(metrics.box.map),
                "f1_from_reported_precision_recall": (
                    float(2 * metrics.box.mp * metrics.box.mr / (metrics.box.mp + metrics.box.mr))
                    if metrics.box.mp + metrics.box.mr
                    else 0.0
                ),
                "validator_speed_ms": {key: float(value) for key, value in metrics.speed.items()},
                "wall_time_seconds": elapsed,
                "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                # YOLO.val owns a validator rather than model.predictor. The CUDA
                # guard above and device=0 make this assignment deterministic.
                "device": "cuda:0",
            }
        )
        print(f"completed resolution={resolution}", flush=True)

    result = {
        "status": "PROVISIONAL_PHONE_COMPONENT_DIAGNOSTIC_COMPLETE",
        "label_status": "PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA",
        "selection_eligible": False,
        "production_claim_allowed": False,
        "independent_test_claim": False,
        "training_performed": False,
        "test_split_accessed": False,
        "scope": "Frozen checkpoint on existing bootstrap validation split only",
        "pre_registered_resolutions": list(RESOLUTIONS),
        "validation_counts": counts,
        "checkpoint_sha256": digest(checkpoint),
        "bootstrap_manifest_sha256": digest(bootstrap_manifest),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "variants": variants,
        "event_metric_status": "NOT_MEASURED_DIFFERENT_TASK_AND_DATASET",
        "development_ai85_status": "DEVELOPMENT_TARGET_NOT_MET",
    }
    output = Path("/kaggle/working/v27_phone_component_eval.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({
        "status": result["status"],
        "gpu": result["gpu"],
        "variants": variants,
    }))


if __name__ == "__main__":
    main()
