"""GPU preflight for the V27 D5 model path; no DMD media or accuracy claim."""

from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def mount(slug: str) -> Path:
    candidates = (
        Path("/kaggle/input") / slug,
        Path("/kaggle/input/datasets/longngoc11") / slug,
    )
    return next((path for path in candidates if path.is_dir()), candidates[0])


def main() -> None:
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.187"],
        check=True,
        timeout=300,
    )
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("GPU_NOT_ALLOCATED")

    code_root = mount("roadwatch-v27-ai85-execution")
    package_root = code_root / "package"
    manifest = json.loads((code_root / "package-manifest.json").read_text())
    for entry in manifest["files"]:
        relative = entry["path"].removeprefix("package/")
        path = package_root / relative
        if not path.is_file() or digest(path) != entry["sha256"]:
            raise ValueError(f"CODE_PACKAGE_HASH_MISMATCH:{relative}")
    sys.path.insert(0, str(package_root))

    from backend.ai.auxiliary import PoseEstimator, PoseEvidence
    from training.dmd.behavior_roi import Box
    from training.dmd.driver_phone_association import DriverPhoneAssociation, OccupantTrack
    from training.dmd.run_autonomous_v21 import (
        evaluate_phone_candidates,
        select_best_phone_candidate,
    )

    pose_evidence = PoseEvidence(
        confidence=0.9,
        face_points=((250.0, 100.0),),
        hand_points=((250.0, 250.0),),
        xyxy=(50.0, 20.0, 700.0, 560.0),
    )
    driver = OccupantTrack(
        "driver_seat_pose_0",
        "driver",
        Box(50.0, 20.0, 700.0, 560.0),
        True,
        0.9,
    )
    detections = [
        {"candidate_id": "static", "local_box": [600, 40, 650, 90],
         "box": [600, 40, 650, 90], "confidence": 0.99, "source": "auxiliary_roi"},
        {"candidate_id": "driver-handheld", "local_box": [235, 235, 270, 280],
         "box": [235, 235, 270, 280], "confidence": 0.55, "source": "specialist_roi"},
    ]
    selected = select_best_phone_candidate(
        evaluate_phone_candidates(
            detections,
            pose_evidence,
            (driver,),
            DriverPhoneAssociation(minimum_containment=0.35, ambiguity_margin=0.15),
            verified_association=True,
        )
    )
    if selected is None or selected["det"]["candidate_id"] != "driver-handheld":
        raise AssertionError("CANDIDATE_IDENTITY_REGRESSION")

    checkpoint_root = mount("roadwatch-ai85-plus-frozen-checkpoints-v8")
    checkpoint_manifest = json.loads(
        (checkpoint_root / "checkpoint_manifest.json").read_text()
    )
    specialist_path = checkpoint_root / "phone_detector.pt"
    if digest(specialist_path) != checkpoint_manifest["phone_detector"]["sha256"]:
        raise ValueError("SPECIALIST_CHECKPOINT_HASH_MISMATCH")

    data_root = mount("roadwatch-mc-bootstrap-6500-v1")
    images = sorted(
        path
        for path in data_root.rglob("*")
        if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        and "/images/val/" in path.as_posix()
    )[:32]
    if len(images) < 32:
        raise ValueError("NEED_32_EXISTING_VALIDATION_IMAGES")

    specialist = YOLO(str(specialist_path))
    auxiliary = YOLO("yolo11n.pt")
    YOLO("yolo11n-pose.pt")
    pose = PoseEstimator(Path("yolo11n-pose.pt"), device=0)
    stage_seconds = {"specialist": [], "auxiliary": [], "pose": []}
    proposal_counts = {"specialist": 0, "auxiliary": 0, "pose": 0}

    for path in images[:3]:
        frame = cv2.imread(str(path))
        specialist.predict(frame, imgsz=640, conf=0.15, device=0, verbose=False)
        auxiliary.predict(frame, imgsz=640, conf=0.15, classes=[67], device=0, verbose=False)
        pose.predict(frame)

    torch.cuda.reset_peak_memory_stats()
    for path in images:
        frame = cv2.imread(str(path))
        if frame is None:
            raise ValueError(f"IMAGE_DECODE_FAILED:{path.name}")
        for name, call in (
            ("specialist", lambda: specialist.predict(
                frame, imgsz=640, conf=0.15, device=0, verbose=False
            )[0]),
            ("auxiliary", lambda: auxiliary.predict(
                frame, imgsz=640, conf=0.15, classes=[67], device=0, verbose=False
            )[0]),
        ):
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = call()
            torch.cuda.synchronize()
            stage_seconds[name].append(time.perf_counter() - started)
            proposal_counts[name] += len(output.boxes)
        torch.cuda.synchronize()
        started = time.perf_counter()
        pose_output = pose.predict(frame)
        torch.cuda.synchronize()
        stage_seconds["pose"].append(time.perf_counter() - started)
        proposal_counts["pose"] += len(pose_output)

    stage_metrics = {
        name: {
            "calls": len(values),
            "p50_ms": float(np.percentile(values, 50) * 1000),
            "p95_ms": float(np.percentile(values, 95) * 1000),
            "mean_ms": float(np.mean(values) * 1000),
        }
        for name, values in stage_seconds.items()
    }
    result = {
        "status": "V27_GPU_PREFLIGHT_PASSED",
        "scope": "32 consumed private validation images; device/runtime check only",
        "training_performed": False,
        "dmd_media_used": False,
        "event_metrics_computed": False,
        "accuracy_claim": None,
        "candidate_identity_test": "PASSED",
        "python": platform.python_version(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0),
        "ultralytics": ultralytics.__version__,
        "opencv": cv2.__version__,
        "device_provenance": {
            "specialist": str(specialist.predictor.device),
            "auxiliary": str(auxiliary.predictor.device),
            "pose": str(pose._model.predictor.device),
        },
        "specialist_sha256": digest(specialist_path),
        "auxiliary_sha256": digest(Path("yolo11n.pt")),
        "pose_sha256": digest(Path("yolo11n-pose.pt")),
        "code_manifest_sha256": digest(code_root / "package-manifest.json"),
        "images": len(images),
        "input_sha256": [digest(path) for path in images],
        "proposal_counts": proposal_counts,
        "stage_runtime": stage_metrics,
        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
        "raw_d5_status": "BLOCKED_NO_AUTHORIZED_DMD_MEDIA_ON_KAGGLE",
        "development_ai85_status": "DEVELOPMENT_TARGET_NOT_MET",
    }
    output = Path("/kaggle/working/v27_gpu_preflight.json")
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({key: result[key] for key in (
        "status", "gpu", "device_provenance", "raw_d5_status"
    )}))


if __name__ == "__main__":
    main()
