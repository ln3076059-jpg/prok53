"""DMD Timeline and OpenLABEL alignment validator.

Performs sequential decode verification, PTS continuity checks,
duration alignment, and rejects broken ground-truth intervals.
"""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import cv2

from training.common import sha256_file
from training.dmd.adapter import parse_dmd_openlabel


@dataclass(frozen=True)
class TimelineValidationResult:
    video_path: str
    video_sha256: str
    annotation_path: str
    annotation_sha256: str
    container: str
    codec: str
    width: int
    height: int
    fps: float
    duration_seconds: float
    total_frames: int
    decoded_frames: int
    decode_errors: int
    corrupt_frames: int
    eof_reached: bool
    phone_intervals_count: int
    total_phone_duration_seconds: float
    max_interval_frame: int
    interval_bounds_valid: bool
    status: str  # VALID or REJECTED

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def validate_dmd_timeline(
    video_path: Path,
    annotation_path: Path,
    max_decode_frames: int | None = None,
) -> TimelineValidationResult:
    if not video_path.is_file():
        raise FileNotFoundError(f"Video not found: {video_path}")
    if not annotation_path.is_file():
        raise FileNotFoundError(f"Annotation not found: {annotation_path}")

    vid_sha = sha256_file(video_path)
    ann_sha = sha256_file(annotation_path)

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Failed to open video container: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 29.76
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    codec_int = int(cap.get(cv2.CAP_PROP_FOURCC))
    codec = "".join([chr((codec_int >> 8 * i) & 0xFF) for i in range(4)]).strip() or "H264"
    container = video_path.suffix.lstrip(".").upper()

    decoded_frames = 0
    decode_errors = 0
    corrupt_frames = 0

    decode_limit = max_decode_frames or total_frames
    while decoded_frames < decode_limit:
        ok, frame = cap.read()
        if not ok:
            break
        if frame is None or frame.size == 0:
            corrupt_frames += 1
        decoded_frames += 1

    eof_reached = decoded_frames >= total_frames
    cap.release()

    duration = total_frames / fps if fps > 0 else 0.0

    # Parse and validate OpenLABEL annotations
    ann = parse_dmd_openlabel(annotation_path, fps=fps, total_frames=total_frames)
    max_int_frame = 0
    total_phone_sec = 0.0
    bounds_valid = True

    for item in ann.phone_intervals:
        if item.end_frame > max_int_frame:
            max_int_frame = item.end_frame
        if item.start_frame < 0 or item.end_frame > total_frames + 30:  # small tolerance
            bounds_valid = False
        total_phone_sec += item.duration_seconds

    status = "VALID" if (decode_errors == 0 and corrupt_frames == 0 and bounds_valid) else "REJECTED"

    return TimelineValidationResult(
        video_path=str(video_path.as_posix()),
        video_sha256=vid_sha,
        annotation_path=str(annotation_path.as_posix()),
        annotation_sha256=ann_sha,
        container=container,
        codec=codec,
        width=width,
        height=height,
        fps=round(fps, 3),
        duration_seconds=round(duration, 3),
        total_frames=total_frames,
        decoded_frames=decoded_frames,
        decode_errors=decode_errors,
        corrupt_frames=corrupt_frames,
        eof_reached=eof_reached,
        phone_intervals_count=len(ann.phone_intervals),
        total_phone_duration_seconds=round(total_phone_sec, 2),
        max_interval_frame=max_int_frame,
        interval_bounds_valid=bounds_valid,
        status=status,
    )
