"""Benchmark direct source crops against the legacy 64px RGB cache path.

This local-only diagnostic measures preprocessing information loss and CPU cost.
It does not train a model and does not use the result as an accuracy metric.
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import psutil
import torch
from torchvision.models import ResNet18_Weights, resnet18

from training.common import sha256_file
from training.dmd.research_permissions import require_operation
from training.dmd.run_v5_experiments import discover_external_dmd_root


ROI_XYXY = (0.0, 0.10, 0.65, 0.95)


def _find_one(directory: Path, pattern: str) -> Path:
    matches = sorted(directory.glob(pattern))
    if len(matches) != 1:
        raise ValueError(f"expected one {pattern} in {directory}, found {len(matches)}")
    return matches[0]


def _decode_crops(
    video_path: Path,
    frame_indices: np.ndarray,
    *,
    output_size: int,
) -> tuple[np.ndarray, float]:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"cannot open video: {video_path}")
    frames = np.empty(
        (len(frame_indices), output_size, output_size, 3), dtype=np.uint8
    )
    current = -1
    started = time.perf_counter()
    for output_index, target in enumerate(frame_indices):
        while current < int(target):
            if not cap.grab():
                cap.release()
                raise ValueError(f"video ended before frame {target}")
            current += 1
        ok, frame = cap.retrieve()
        if not ok or frame is None:
            cap.release()
            raise ValueError(f"cannot decode frame {target}")
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = ROI_XYXY
        crop = frame[
            max(0, int(y1 * height)) : min(height, int(y2 * height)),
            max(0, int(x1 * width)) : min(width, int(x2 * width)),
        ]
        resized = cv2.resize(
            crop, (output_size, output_size), interpolation=cv2.INTER_AREA
        )
        frames[output_index] = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    cap.release()
    return frames, time.perf_counter() - started


def _legacy_upscale(frames_64: np.ndarray, size: int = 112) -> np.ndarray:
    return np.stack(
        [
            cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
            for frame in frames_64
        ]
    )


def _sharpness(frames: np.ndarray) -> dict[str, float]:
    values = [
        float(cv2.Laplacian(cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY), cv2.CV_64F).var())
        for frame in frames
    ]
    return {
        "mean_laplacian_variance": round(float(np.mean(values)), 4),
        "median_laplacian_variance": round(float(np.median(values)), 4),
    }


def _encode(
    model: torch.nn.Module,
    frames: np.ndarray,
    *,
    batch_size: int,
) -> tuple[np.ndarray, float, int]:
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    process = psutil.Process()
    peak_rss = process.memory_info().rss
    encoded: list[np.ndarray] = []
    started = time.perf_counter()
    with torch.inference_mode():
        for start in range(0, len(frames), batch_size):
            tensor = torch.from_numpy(frames[start : start + batch_size].copy())
            tensor = tensor.permute(0, 3, 1, 2).float().div_(255.0)
            tensor = (tensor - mean) / std
            encoded.append(model(tensor).cpu().numpy().astype(np.float32))
            peak_rss = max(peak_rss, process.memory_info().rss)
    elapsed = time.perf_counter() - started
    return np.concatenate(encoded), elapsed, peak_rss


def _cosine_summary(left: np.ndarray, right: np.ndarray) -> dict[str, float]:
    denominator = np.linalg.norm(left, axis=1) * np.linalg.norm(right, axis=1)
    values = np.sum(left * right, axis=1) / np.maximum(denominator, 1e-12)
    return {
        "mean": round(float(np.mean(values)), 6),
        "median": round(float(np.median(values)), 6),
        "minimum": round(float(np.min(values)), 6),
    }


def run(
    *,
    subject: str,
    maximum_samples: int,
    batch_size: int,
    repeats: int,
    output_path: Path,
) -> dict[str, Any]:
    require_operation("local_dmd_training")
    if maximum_samples < 1 or batch_size < 1 or repeats < 1:
        raise ValueError("maximum_samples, batch_size, and repeats must be positive")
    root = discover_external_dmd_root()
    feature_path = root / "derivatives" / f"v3_features_{subject}.json"
    feature = json.loads(feature_path.read_text(encoding="utf-8"))
    records = feature["records"][:maximum_samples]
    frame_indices = np.asarray(
        [int(row["frame_index"]) for row in records], dtype=np.int64
    )
    video_path = _find_one(root / "subjects" / subject, "*_rgb_body.mp4")
    torch.set_num_threads(4)
    weights = ResNet18_Weights.DEFAULT
    model = resnet18(weights=weights)
    model.fc = torch.nn.Identity()
    model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    direct_64, decode_64 = _decode_crops(video_path, frame_indices, output_size=64)
    legacy_112 = _legacy_upscale(direct_64, 112)
    direct_112, decode_112 = _decode_crops(video_path, frame_indices, output_size=112)
    direct_160, decode_160 = _decode_crops(video_path, frame_indices, output_size=160)

    variants = {
        "LEGACY_64_TO_112": legacy_112,
        "DIRECT_112": direct_112,
        "DIRECT_160": direct_160,
    }
    embeddings: dict[str, np.ndarray] = {}
    measurements: dict[str, Any] = {}
    _encode(model, legacy_112[: min(8, len(legacy_112))], batch_size=batch_size)
    for name, frames in variants.items():
        repeat_times: list[float] = []
        repeat_peaks: list[int] = []
        encoded = np.empty((0, 512), dtype=np.float32)
        for _ in range(repeats):
            encoded, encode_seconds, peak_rss = _encode(
                model, frames, batch_size=batch_size
            )
            repeat_times.append(encode_seconds)
            repeat_peaks.append(peak_rss)
        median_seconds = float(np.median(repeat_times))
        embeddings[name] = encoded
        measurements[name] = {
            "input_shape": list(frames.shape),
            "sharpness": _sharpness(frames),
            "encode_repeats_seconds": [round(value, 4) for value in repeat_times],
            "encode_median_seconds": round(median_seconds, 4),
            "encode_median_frames_per_second": round(
                len(frames) / max(median_seconds, 1e-9), 4
            ),
            "maximum_process_rss_bytes_across_repeats": max(repeat_peaks),
        }
    measurements["LEGACY_64_TO_112"]["decode_seconds"] = round(decode_64, 4)
    measurements["DIRECT_112"]["decode_seconds"] = round(decode_112, 4)
    measurements["DIRECT_160"]["decode_seconds"] = round(decode_160, 4)

    weights_path = Path(torch.hub.get_dir()) / "checkpoints" / Path(weights.url).name
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "COMPLETED_DIAGNOSTIC_ONLY",
        "subject": subject,
        "samples": len(frame_indices),
        "timing_repeats_after_warmup": repeats,
        "sample_stride": feature["sample_stride"],
        "covered_seconds": round(float(records[-1]["timestamp_seconds"]), 4),
        "roi_xyxy": list(ROI_XYXY),
        "encoder": "TORCHVISION_RESNET18_IMAGENET1K_V1_FROZEN",
        "encoder_weights_sha256": sha256_file(weights_path),
        "measurements": measurements,
        "embedding_cosine": {
            "legacy_vs_direct112": _cosine_summary(
                embeddings["LEGACY_64_TO_112"], embeddings["DIRECT_112"]
            ),
            "direct112_vs_direct160": _cosine_summary(
                embeddings["DIRECT_112"], embeddings["DIRECT_160"]
            ),
        },
        "interpretation_guard": (
            "Sharpness, cosine distance, and throughput are preprocessing/runtime "
            "diagnostics only; model selection requires subject-disjoint task metrics. "
            "Laplacian variance is only directly comparable at the same output size."
        ),
        "governance": {
            "execution_location": "LOCAL_ONLY",
            "dmd_media_uploaded": False,
            "training_performed": False,
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject", default="gC-14")
    parser.add_argument("--maximum-samples", type=int, default=120)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            "reports/v31_ai85_plan_execution/direct_crop_benchmark_gC14.json"
        ),
    )
    args = parser.parse_args()
    result = run(
        subject=args.subject,
        maximum_samples=args.maximum_samples,
        batch_size=args.batch_size,
        repeats=args.repeats,
        output_path=args.output,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
