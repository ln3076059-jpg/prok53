from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import cv2
import yaml

from backend.ai.auxiliary import crop_box
from backend.ai.detector import SafetyDetector
from backend.services.video_analyzer import VideoAnalyzer


def _summary(values: list[float]) -> dict:
    if not values:
        return {"status": "NOT_RUN", "p50_ms": None, "p95_ms": None}
    ordered = sorted(values)
    p95_index = min(len(ordered) - 1, max(0, int(round(0.95 * (len(ordered) - 1)))))
    return {
        "status": "MEASURED",
        "samples": len(values),
        "p50_ms": statistics.median(values),
        "p95_ms": ordered[p95_index],
    }


def benchmark(video: Path, config_path: Path, frames: int, evidence_root: Path) -> dict:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    detector = SafetyDetector(Path(config.get("weights", "models/active/best.pt")), config_path)
    analyzer = VideoAnalyzer(detector, evidence_root)
    capture = cv2.VideoCapture(str(video))
    if not capture.isOpened():
        raise ValueError(f"cannot open benchmark video: {video}")

    decode_latencies: list[float] = []
    vehicle_latencies: list[float] = []
    behavior_latencies: list[float] = []
    pose_latencies: list[float] = []
    classifier_latencies: list[float] = []
    total_latencies: list[float] = []
    decoded = []
    try:
        for _ in range(frames):
            started = time.perf_counter()
            ok, frame = capture.read()
            decode_ms = (time.perf_counter() - started) * 1000
            if not ok:
                break
            decode_latencies.append(decode_ms)
            decoded.append(frame)
    finally:
        capture.release()

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        torch = None

    for frame in decoded:
        total_started = time.perf_counter()
        if analyzer.vehicle_detector.available:
            started = time.perf_counter()
            analyzer.vehicle_detector.predict(frame)
            vehicle_latencies.append((time.perf_counter() - started) * 1000)
        detections = []
        if detector.available:
            started = time.perf_counter()
            detections = detector.predict(frame, track=False)
            behavior_latencies.append((time.perf_counter() - started) * 1000)
        if analyzer.pose.available:
            started = time.perf_counter()
            analyzer.pose.predict(frame)
            pose_latencies.append((time.perf_counter() - started) * 1000)
        upper_body = next(
            (item for item in detections if item.class_name == "occupant_upper_body"), None
        )
        if analyzer.seatbelt_classifier.available and upper_body is not None:
            started = time.perf_counter()
            analyzer.seatbelt_classifier.predict(crop_box(frame, upper_body.xyxy))
            classifier_latencies.append((time.perf_counter() - started) * 1000)
        total_latencies.append((time.perf_counter() - total_started) * 1000)

    decode_seconds = sum(decode_latencies) / 1000
    serial_latencies = [
        decode_ms + inference_ms
        for decode_ms, inference_ms in zip(decode_latencies, total_latencies, strict=True)
    ]
    serial_seconds = sum(serial_latencies) / 1000
    report = {
        "status": "MEASURED" if decoded else "NOT_RUN",
        "video": str(video),
        "frames": len(decoded),
        "decode_fps": len(decoded) / decode_seconds if decode_seconds > 0 else None,
        "decode_latency": _summary(decode_latencies),
        "vehicle_detector_latency": _summary(vehicle_latencies),
        "behavior_detector_latency": _summary(behavior_latencies),
        "pose_latency": _summary(pose_latencies),
        "classifier_latency": _summary(classifier_latencies),
        "component_inference_latency": _summary(total_latencies),
        "serial_decode_plus_component_latency": _summary(serial_latencies),
        "serial_component_fps": len(decoded) / serial_seconds if serial_seconds > 0 else None,
        "measurement_scope": (
            "Decoded-frame component benchmark; excludes database writes, evidence encoding, "
            "queue latency, and network transport."
        ),
        "gpu_vram_peak_mb": None,
        "cpu_ram_mb": None,
    }
    if torch is not None and torch.cuda.is_available():
        report["gpu_vram_peak_mb"] = torch.cuda.max_memory_allocated() / (1024 * 1024)
    try:
        import os

        import psutil

        report["cpu_ram_mb"] = psutil.Process(os.getpid()).memory_info().rss / (1024 * 1024)
    except ImportError:
        report["cpu_ram_status"] = "NOT_RUN_PSUTIL_NOT_INSTALLED"
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure Roadwatch runtime latency on real decoded frames"
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--config", type=Path, default=Path("models/model_config_v2.yaml"))
    parser.add_argument("--frames", type=int, default=100)
    parser.add_argument("--evidence-root", type=Path, default=Path("evidence/benchmark"))
    parser.add_argument("--output", type=Path, default=Path("reports/runtime_benchmark.json"))
    args = parser.parse_args()
    report = (
        benchmark(args.video, args.config, args.frames, args.evidence_root)
        if args.video
        else {"status": "NOT_RUN", "reason": "pass --video to measure real runtime"}
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
