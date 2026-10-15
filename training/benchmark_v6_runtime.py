"""Measure Roadwatch V6 runtime components and a real cabin-video backend pipeline."""

from __future__ import annotations

import argparse
import json
import os
import tempfile
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import sessionmaker

from backend.ai.auxiliary import SeatbeltClassifier
from backend.ai.detector import SafetyDetector
from backend.ai.fusion import FusionFeatures, LogisticFusion
from backend.database import Base
from backend.models.entities import AnalysisJob, Detection, Event, InputScope, Video
from backend.services.video_analyzer import VideoAnalyzer
from training.common import sha256_file
from training.dmd.run_v5_experiments import discover_external_dmd_root
from training.dmd.v5_resnet_gru import (
    ResNetGRUClassifier,
    SubjectEmbeddings,
    extract_resnet18_embeddings,
    predict_embeddings,
)


def _summary(values: list[float]) -> dict[str, float | int | None]:
    if not values:
        return {"count": 0, "mean_ms": None, "p50_ms": None, "p95_ms": None}
    return {
        "count": len(values),
        "mean_ms": round(float(np.mean(values)), 4),
        "p50_ms": round(float(np.percentile(values, 50)), 4),
        "p95_ms": round(float(np.percentile(values, 95)), 4),
    }


def _timed_wrapper(function: Callable[..., Any], samples: list[float]) -> Callable[..., Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        started = time.perf_counter()
        try:
            return function(*args, **kwargs)
        finally:
            samples.append((time.perf_counter() - started) * 1000.0)

    return wrapper


def _create_clip(source: Path, destination: Path, maximum_frames: int) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source))
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 25.0)
    width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
    writer = cv2.VideoWriter(
        str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height)
    )
    if not writer.isOpened():
        raise RuntimeError("could not create temporary benchmark video")
    count = 0
    try:
        while count < maximum_frames:
            ok, frame = capture.read()
            if not ok:
                break
            writer.write(frame)
            count += 1
    finally:
        capture.release()
        writer.release()
    return {"frames": count, "fps": fps, "width": width, "height": height}


def _decode_benchmark(source: Path, maximum_frames: int = 300) -> dict[str, Any]:
    capture = cv2.VideoCapture(str(source))
    samples: list[float] = []
    started = time.perf_counter()
    try:
        for _ in range(maximum_frames):
            frame_started = time.perf_counter()
            ok, _frame = capture.read()
            if not ok:
                break
            samples.append((time.perf_counter() - frame_started) * 1000.0)
    finally:
        capture.release()
    elapsed = time.perf_counter() - started
    return {
        **_summary(samples),
        "elapsed_seconds": round(elapsed, 4),
        "decode_fps": round(len(samples) / elapsed, 3) if elapsed else None,
    }


def _temporal_benchmark(checkpoint_path: Path, embeddings_path: Path) -> dict[str, Any]:
    payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if payload["variant"]["architecture"] != "flat":
        return {"status": "NOT_RUN_FOR_NON_FLAT_CHECKPOINT"}
    model = ResNetGRUClassifier()
    model.load_state_dict(payload["model_state_dict"])
    subject_id = str(payload["heldout_subject"])
    subject = SubjectEmbeddings.load(subject_id, embeddings_path)
    result = predict_embeddings(model, subject, sequence_length=9, batch_size=16)
    return {
        "status": "MEASURED",
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": sha256_file(checkpoint_path),
        "clips": len(result["centers"]),
        "clips_per_second": result["clips_per_second"],
        "milliseconds_per_clip": result["milliseconds_per_clip"],
        "p50_latency_ms": result["p50_latency_ms"],
        "p95_latency_ms": result["p95_latency_ms"],
    }


def run(config_path: Path, output_path: Path, frames: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    process = psutil.Process(os.getpid())
    dmd_root = discover_external_dmd_root()
    source = next((dmd_root / "subjects" / "gB-9").glob("*_rgb_body.mp4"))
    decode = _decode_benchmark(source)
    memory_samples = [process.memory_info().rss / (1024 * 1024)]
    stop_memory = threading.Event()

    def sample_memory() -> None:
        while not stop_memory.wait(0.02):
            memory_samples.append(process.memory_info().rss / (1024 * 1024))

    sampler = threading.Thread(target=sample_memory, daemon=True)
    sampler.start()
    detector_samples: list[float] = []
    seatbelt_samples: list[float] = []
    fusion_samples: list[float] = []
    context_samples: list[float] = []
    evidence_samples: list[float] = []

    try:
        with tempfile.TemporaryDirectory(prefix="roadwatch-v6-benchmark-") as temporary:
            temp_root = Path(temporary)
            clip_path = temp_root / "cabin_clip.mp4"
            clip = _create_clip(source, clip_path, frames)
            engine = create_engine(f"sqlite:///{(temp_root / 'benchmark.db').as_posix()}")
            Base.metadata.create_all(engine)
            Session = sessionmaker(bind=engine, expire_on_commit=False)
            session = Session()
            video = Video(
                original_name=clip_path.name,
                storage_path=str(clip_path),
                sha256=sha256_file(clip_path),
                size_bytes=clip_path.stat().st_size,
                mime_type="video/mp4",
                input_scope=InputScope.VEHICLE_CABIN_CROP,
            )
            session.add(video)
            session.commit()
            job = AnalysisJob(video_id=video.id)
            session.add(job)
            session.commit()

            detector = SafetyDetector(Path("models/active/best.pt"), config_path)
            detector.validate_runtime("development")
            analyzer = VideoAnalyzer(detector, temp_root / "evidence")
            detector.predict = _timed_wrapper(detector.predict, detector_samples)
            analyzer.seatbelt_classifier.predict = _timed_wrapper(
                analyzer.seatbelt_classifier.predict, seatbelt_samples
            )
            analyzer.fusion.predict = _timed_wrapper(analyzer.fusion.predict, fusion_samples)
            analyzer._analyze_context = _timed_wrapper(  # type: ignore[method-assign]
                analyzer._analyze_context, context_samples
            )
            analyzer._persist_artifacts = _timed_wrapper(  # type: ignore[method-assign]
                analyzer._persist_artifacts, evidence_samples
            )
            started = time.perf_counter()
            analyzer.analyze(session, job, video)
            pipeline_seconds = time.perf_counter() - started
            detection_count = session.scalar(select(func.count()).select_from(Detection)) or 0
            event_count = session.scalar(select(func.count()).select_from(Event)) or 0
            session.close()
            engine.dispose()

            classifier = SeatbeltClassifier(Path("models/active/v2/seatbelt_classifier.pt"))
            capture = cv2.VideoCapture(str(clip_path))
            direct_seatbelt: list[float] = []
            for _ in range(min(frames, 20)):
                ok, frame = capture.read()
                if not ok:
                    break
                started = time.perf_counter()
                classifier.predict(frame)
                direct_seatbelt.append((time.perf_counter() - started) * 1000.0)
            capture.release()

            fusion = LogisticFusion(Path("models/fusion/no_seatbelt_v2.json"), 0.25)
            fusion_direct: list[float] = []
            features = FusionFeatures(
                detector_confidence=0.7,
                classifier_unfastened=0.65,
                classifier_fastened=0.2,
                pose_confidence=0.8,
                temporal_ratio=0.6,
                driver_role=1.0,
                vehicle_context=1.0,
            )
            for _ in range(1000):
                started = time.perf_counter()
                fusion.predict(features)
                fusion_direct.append((time.perf_counter() - started) * 1000.0)

            image_write: list[float] = []
            capture = cv2.VideoCapture(str(clip_path))
            for index in range(min(frames, 20)):
                ok, frame = capture.read()
                if not ok:
                    break
                started = time.perf_counter()
                cv2.imwrite(str(temp_root / f"evidence_probe_{index:03d}.jpg"), frame)
                image_write.append((time.perf_counter() - started) * 1000.0)
            capture.release()

            rgb_cache = dmd_root / "derivatives" / "v5_rgb_cache" / "v5_rgb_gB-9_64.npz"
            embedding_output = temp_root / "resnet18_embeddings.npz"
            embedding = extract_resnet18_embeddings(
                "gB-9", rgb_cache, embedding_output, batch_size=16
            )
            checkpoint = Path(
                "models/candidate/v6_action_ablation_001/flat_unweighted_loso_gB-9.pt"
            )
            temporal = _temporal_benchmark(
                checkpoint,
                dmd_root / "derivatives" / "v5_resnet18_embeddings" / "v5_resnet18_gB-9.npz",
            )
    finally:
        stop_memory.set()
        sampler.join()
        memory_samples.append(process.memory_info().rss / (1024 * 1024))

    cuda_available = torch.cuda.is_available()
    result = {
        "status": "MEASURED_LOCAL_CPU",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "hardware": {
            "logical_cpu_count": psutil.cpu_count(logical=True),
            "physical_cpu_count": psutil.cpu_count(logical=False),
            "total_ram_mb": round(psutil.virtual_memory().total / (1024 * 1024), 1),
            "cuda_available": cuda_available,
            "vram_peak_mb": (
                round(torch.cuda.max_memory_allocated() / (1024 * 1024), 1)
                if cuda_available
                else 0.0
            ),
        },
        "source": {
            "path": str(source),
            "sha256": sha256_file(source),
            "scope": "AUTHORIZED_DMD_CABIN_BODY_VIDEO; DEVELOPMENT_SUBJECT_gB-9",
        },
        "video_decode": decode,
        "detector": _summary(detector_samples),
        "seatbelt_classifier_in_pipeline": _summary(seatbelt_samples),
        "seatbelt_classifier_direct_full_frame_probe": _summary(direct_seatbelt),
        "fusion_in_pipeline": _summary(fusion_samples),
        "fusion_direct": _summary(fusion_direct),
        "evidence_persist_calls": _summary(evidence_samples),
        "evidence_jpeg_write": _summary(image_write),
        "resnet_embedding": {
            **embedding,
            "frames_per_second": round(
                embedding["records"] / embedding["extraction_seconds"], 3
            ),
        },
        "temporal_head": temporal,
        "full_backend_pipeline": {
            "frames": clip["frames"],
            "source_fps": clip["fps"],
            "processing_seconds": round(pipeline_seconds, 4),
            "processing_fps": round(clip["frames"] / pipeline_seconds, 3),
            "real_time_factor": round(pipeline_seconds / (clip["frames"] / clip["fps"]), 3),
            "per_context_latency": _summary(context_samples),
            "database_detection_rows": int(detection_count),
            "database_event_rows": int(event_count),
            "scope": "ACTUAL VideoAnalyzer + SQLite + evidence path on a cabin clip",
        },
        "memory": {
            "rss_start_mb": round(memory_samples[0], 1),
            "rss_peak_mb": round(max(memory_samples), 1),
            "rss_end_mb": round(memory_samples[-1], 1),
        },
        "interpretation": (
            "This is a local cabin-video benchmark, not roadside accuracy evidence. "
            "Cached temporal-head throughput is reported separately from full-pipeline FPS."
        ),
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config", type=Path, default=Path("models/model_config_v2.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v6_completion/v6_runtime_benchmark.json"),
    )
    parser.add_argument("--frames", type=int, default=60)
    args = parser.parse_args()
    result = run(args.config, args.output, args.frames)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
