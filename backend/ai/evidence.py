from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def evidence_package_sha256(hashes: dict[str, str]) -> str:
    payload = json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


@dataclass(frozen=True)
class EvidenceArtifacts:
    event_id: str
    original_path: Path
    annotated_path: Path
    clip_path: Path | None
    trace_path: Path
    hashes: dict[str, str]


@dataclass
class _PendingEvidence:
    event_id: str
    root: Path
    key_frame: np.ndarray
    annotated_frame: np.ndarray
    frames: list[tuple[float, np.ndarray]]
    end_timestamp: float
    trace: dict


class EventEvidenceBuffer:
    """Keep pre-event frames and finalize a multi-frame evidence package after the event."""

    def __init__(self, fps: float, pre_seconds: float = 2.0, post_seconds: float = 2.5):
        self.fps = max(float(fps), 1.0)
        self.pre_seconds = max(float(pre_seconds), 0.0)
        self.post_seconds = max(float(post_seconds), 0.0)
        capacity = max(1, int(self.fps * self.pre_seconds) + 1)
        self.frames: deque[tuple[float, np.ndarray]] = deque(maxlen=capacity)
        self.pending: dict[str, _PendingEvidence] = {}

    def add_frame(self, timestamp: float, frame: np.ndarray) -> list[EvidenceArtifacts]:
        copied = frame.copy()
        self.frames.append((float(timestamp), copied))
        ready: list[EvidenceArtifacts] = []
        for event_id, item in list(self.pending.items()):
            if timestamp <= item.end_timestamp:
                item.frames.append((float(timestamp), copied.copy()))
            if timestamp >= item.end_timestamp:
                ready.append(self._write(item))
                del self.pending[event_id]
        return ready

    def register(
        self,
        event_id: str,
        evidence_root: Path,
        timestamp: float,
        key_frame: np.ndarray,
        annotated_frame: np.ndarray,
        trace: dict,
    ) -> None:
        if event_id in self.pending:
            raise ValueError(f"evidence already pending for event {event_id}")
        root = evidence_root / event_id
        if root.exists():
            raise FileExistsError(f"evidence directory already exists: {root}")
        pre_frames = [(ts, image.copy()) for ts, image in self.frames if ts <= timestamp]
        self.pending[event_id] = _PendingEvidence(
            event_id=event_id,
            root=root,
            key_frame=key_frame.copy(),
            annotated_frame=annotated_frame.copy(),
            frames=pre_frames,
            end_timestamp=float(timestamp) + self.post_seconds,
            trace=dict(trace),
        )

    def flush(self) -> list[EvidenceArtifacts]:
        artifacts = [self._write(item) for item in self.pending.values()]
        self.pending.clear()
        return artifacts

    def _write(self, item: _PendingEvidence) -> EvidenceArtifacts:
        item.root.mkdir(parents=True, exist_ok=False)
        original = item.root / "original_keyframe.jpg"
        annotated = item.root / "annotated_keyframe.jpg"
        if not cv2.imwrite(str(original), item.key_frame):
            raise OSError(f"failed to write {original}")
        if not cv2.imwrite(str(annotated), item.annotated_frame):
            raise OSError(f"failed to write {annotated}")

        clip = item.root / "evidence.mp4"
        clip_path: Path | None = clip if self._write_clip(clip, item.frames) else None
        hashes = {
            original.name: sha256_file(original),
            annotated.name: sha256_file(annotated),
        }
        if clip_path:
            hashes[clip_path.name] = sha256_file(clip_path)

        trace = dict(item.trace)
        actual_start = item.frames[0][0] if item.frames else None
        actual_end = item.frames[-1][0] if item.frames else None
        post_complete = actual_end is not None and actual_end >= item.end_timestamp
        evidence_status = (
            "COMPLETE"
            if clip_path and post_complete
            else "CLIP_TRUNCATED"
            if clip_path
            else "CLIP_UNAVAILABLE"
        )
        trace.update(
            {
                "evidence_status": evidence_status,
                "clip_pre_seconds_requested": self.pre_seconds,
                "clip_post_seconds_requested": self.post_seconds,
                "clip_frame_count": len(item.frames),
                "clip_start_timestamp": actual_start,
                "clip_end_timestamp": actual_end,
                "post_context_complete": post_complete,
                "files": hashes,
            }
        )
        trace_path = item.root / "trace.json"
        trace_path.write_text(json.dumps(trace, indent=2, sort_keys=True), encoding="utf-8")
        hashes[trace_path.name] = sha256_file(trace_path)
        return EvidenceArtifacts(item.event_id, original, annotated, clip_path, trace_path, hashes)

    def _write_clip(self, path: Path, frames: list[tuple[float, np.ndarray]]) -> bool:
        if not frames:
            return False
        height, width = frames[0][1].shape[:2]
        writer = cv2.VideoWriter(
            str(path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (width, height)
        )
        if not writer.isOpened():
            return False
        try:
            for _, frame in frames:
                if frame.shape[:2] != (height, width):
                    frame = cv2.resize(frame, (width, height))
                writer.write(frame)
        finally:
            writer.release()
        return path.is_file() and path.stat().st_size > 0


def annotate_frame(frame: np.ndarray, detections: list) -> np.ndarray:
    marked = frame.copy()
    for detection in detections:
        x1, y1, x2, y2 = map(int, detection.xyxy)
        cv2.rectangle(marked, (x1, y1), (x2, y2), (41, 123, 74), 2)
        cv2.putText(
            marked,
            f"{detection.class_name} {detection.confidence:.2f}",
            (x1, max(18, y1 - 6)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (41, 123, 74),
            1,
            cv2.LINE_AA,
        )
    return marked
