"""Frozen Roadwatch GPU throughput probe; no training or independent accuracy claim."""
from __future__ import annotations

import hashlib
import json
import platform
import subprocess
import sys
import time
from pathlib import Path


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def run():
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "ultralytics==8.3.187"],
        check=True,
        timeout=240,
    )
    import cv2
    import numpy as np
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise RuntimeError("GPU not allocated; refusing to call CPU throughput a GPU result")
    def mount(slug):
        candidates = [Path("/kaggle/input") / slug,
                      Path("/kaggle/input/datasets/longngoc11") / slug]
        return next((p for p in candidates if p.is_dir()), candidates[0])

    checkpoint_root = mount("roadwatch-ai85-plus-frozen-checkpoints-v8")
    manifest = json.loads((checkpoint_root / "checkpoint_manifest.json").read_text())
    weights = checkpoint_root / "phone_detector.pt"
    if digest(weights) != manifest["phone_detector"]["sha256"]:
        raise ValueError("uploaded active phone checkpoint hash mismatch")
    data_root = mount("roadwatch-mc-bootstrap-6500-v1")
    images = sorted(p for p in data_root.rglob("*") if p.suffix.lower() in {".jpg", ".png"}
                    and "/images/val/" in p.as_posix())[:64]
    if len(images) < 64:
        raise ValueError("need 64 images from the existing validation split; no test fallback")
    if any(any(key in p.as_posix().lower() for key in ("ge-28", "ge_28", "gz-37", "gz_37"))
           for p in images):
        raise ValueError("protected subject in requested inference input")
    model = YOLO(str(weights))
    if list(model.names.values()) != ["phone"]:
        raise ValueError("incompatible class mapping")
    configs = [("original_960", 960, False, False), ("resolution_640", 640, False, False),
               ("TTA_640", 640, True, False), ("resolution_1024", 1024, False, False),
               ("agnostic_640", 640, False, True)]
    results = []
    for name, size, tta, agnostic in configs:
        for warmup in images[:3]:
            model.predict(str(warmup), imgsz=size, augment=tta, agnostic_nms=agnostic,
                          conf=.1, iou=.7, device=0, verbose=False)
        records, latencies = [], []
        for path in images:
            # Decode included in end-to-end detector call latency; no temporal model here.
            torch.cuda.synchronize()
            started = time.perf_counter()
            output = model.predict(str(path), imgsz=size, augment=tta, agnostic_nms=agnostic,
                                   conf=.1, iou=.7, device=0, verbose=False)[0]
            torch.cuda.synchronize()
            latencies.append(time.perf_counter() - started)
            records.append({"image": path.relative_to(data_root).as_posix(),
                            "boxes_original_xyxy": output.boxes.xyxy.cpu().tolist(),
                            "confidences": output.boxes.conf.cpu().tolist()})
        results.append({"id": name, "imgsz": size, "TTA": tta, "agnostic_nms": agnostic,
                        "images": len(images), "warmup_images_excluded": 3,
                        "p50_ms": float(np.percentile(latencies, 50) * 1000),
                        "p95_ms": float(np.percentile(latencies, 95) * 1000),
                        "detector_call_fps": float(len(images) / sum(latencies)),
                        "prediction_seconds": latencies, "records": records})
        print(f"GPU probe completed {name}", flush=True)
    result = {"status": "GPU_FROZEN_INFERENCE_MEASURED", "training_performed": False,
              "scope": "64 legacy consumed validation images, not independent accuracy or full-system FPS",
              "dataset": "longngoc11/roadwatch-mc-bootstrap-6500-v1",
              "labels_used": False, "holdouts_accessed": False,
              "weights_sha256": digest(weights), "source_git_head": manifest["git_head"],
              "python": platform.python_version(), "torch": torch.__version__,
              "ultralytics": ultralytics.__version__, "opencv": cv2.__version__,
              "gpu": torch.cuda.get_device_name(0), "cuda": torch.version.cuda,
              "gpu_total_memory_bytes": torch.cuda.get_device_properties(0).total_memory,
              "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
              "input_sha256": [{"path": p.relative_to(data_root).as_posix(), "sha256": digest(p)} for p in images],
              "variants": results, "AP50": None, "AP50_95": None,
              "production_approved": False, "independent_status": "INDEPENDENT_AI85_NOT_ESTABLISHED"}
    Path("/kaggle/working/gpu_frozen_results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: result[k] for k in ("status", "gpu", "ultralytics", "weights_sha256")}))


if __name__ == "__main__":
    run()
