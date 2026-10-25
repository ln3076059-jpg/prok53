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
        h, w = frame.shape[:2]
        if w > 640:
            new_w = 640
            new_h = max(2, int(round(h * (640.0 / w))))
            clip_frame = cv2.resize(frame, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            clip_frame = frame
        self.frames.append((float(timestamp), clip_frame))
        ready: list[EvidenceArtifacts] = []
        for event_id, item in list(self.pending.items()):
            if timestamp <= item.end_timestamp:
                item.frames.append((float(timestamp), clip_frame))
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
        evidence_base = evidence_root.resolve()
        root = (evidence_base / event_id).resolve()
        if evidence_base not in root.parents:
            raise ValueError("event_id escapes the evidence root")
        if root.exists():
            raise FileExistsError(f"evidence directory already exists: {root}")
        pre_frames = [(ts, image) for ts, image in self.frames if ts <= timestamp]
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
    h, w = marked.shape[:2]
    placed_banners: list[tuple[int, int, int, int]] = []
    phone_reserved: list[tuple[int, int, int, int]] = []
    torso_boxes: list[tuple[int, int, int, int]] = []
    for det in detections:
        cname_d = str(getattr(det, "class_name", "") or "").lower()
        px1, py1, px2, py2 = map(int, det.xyxy)
        if "phone" in cname_d:
            reserve_pad = 180 if w >= 900 else 120
            reserve_h = 36 if w >= 900 else 26
            phone_reserved.append((max(2, px1 - 6), max(2, py1 - reserve_h), min(w - 2, px2 + reserve_pad), py2 + 6))
        else:
            torso_boxes.append((px1, py1, px2, py2))
    torso_boxes.sort(key=lambda b: b[0])

    dets_sorted = sorted(
        detections,
        key=lambda d: (
            1 if "phone" in str(getattr(d, "class_name", "") or "").lower() else 0,
            float(d.xyxy[0]),
        ),
    )

    torso_idx = 0
    for idx_d, detection in enumerate(dets_sorted):
        x1, y1, x2, y2 = map(int, detection.xyxy)
        x1 = max(2, min(w - 4, x1))
        y1 = max(2, min(h - 4, y1))
        x2 = max(x1 + 4, min(w - 2, x2))
        y2 = max(y1 + 4, min(h - 2, y2))

        cname = str(getattr(detection, "class_name", "") or "").lower()
        role = str(getattr(detection, "occupant_role", None) or "driver").lower()
        is_driver = role == "driver"
        role_prefix = "DRIVER" if is_driver else "PASSENGER"
        is_phone = "phone" in cname
        is_unfast = "unfastened" in cname or "no_seatbelt" in cname

        if is_phone and is_driver:
            color = (40, 30, 218)  # Red (VI PHAM)
            tag = f"{role_prefix} - PHONE [VI PHAM]"
        elif is_phone and not is_driver:
            color = (195, 114, 0)  # Blue (KHONG VI PHAM)
            tag = f"{role_prefix} - PHONE [KHONG VI PHAM]"
        elif is_unfast:
            color = (43, 131, 255)  # Orange (VI PHAM)
            tag = f"{role_prefix} - NO_SEATBELT [VI PHAM]"
        else:
            color = (72, 161, 36)  # Green (OK)
            tag = f"{role_prefix} - SEATBELT [OK]"

        conf_pct = int(round(float(getattr(detection, "confidence", 0.9)) * 100))
        label = f"{tag} {conf_pct}%"
        font_scale = 0.56 if w >= 900 else 0.40
        thickness = 2 if w >= 900 else 1
        cv2.rectangle(marked, (x1, y1), (x2, y2), color, 3 if w >= 900 else 2)
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        bh = th + 10
        bw = tw + 12
        lx = max(4, min(w - bw - 4, x1))
        if not is_phone and len(torso_boxes) >= 2 and torso_idx == 0:
            right_x1 = torso_boxes[1][0]
            if lx + bw > right_x1 - 4:
                lx = max(4, min(x1, right_x1 - bw - 6))
        top_y = max(2, y1 - bh)
        bot_y = max(bh, min(h - 2 - bh, y2 - bh - 2))
        below_y = min(h - 2 - bh, y2 + 2)
        if not is_phone and len(torso_boxes) >= 2 and torso_idx == 1:
            candidates = [
                (lx, top_y, lx + bw, top_y + bh),
                (lx, bot_y, lx + bw, bot_y + bh),
                (lx, below_y, lx + bw, below_y + bh),
            ]
        else:
            candidates = [
                (lx, top_y, lx + bw, top_y + bh),
                (lx, below_y, lx + bw, below_y + bh),
                (lx, bot_y, lx + bw, bot_y + bh),
            ]
        if not is_phone:
            torso_idx += 1

        obstacles = placed_banners if is_phone else (placed_banners + phone_reserved)
        chosen = candidates[0]
        for cx1, cy1, cx2, cy2 in candidates:
            overlaps = any(
                not (cx2 + 6 <= ox1 or cx1 - 6 >= ox2 or cy2 + 4 <= oy1 or cy1 - 4 >= oy2)
                for ox1, oy1, ox2, oy2 in obstacles
            )
            if not overlaps:
                chosen = (cx1, cy1, cx2, cy2)
                break
        placed_banners.append(chosen)
        bx1, by1, bx2, by2 = chosen
        cv2.rectangle(marked, (bx1, by1), (bx2, by2), color, -1)
        cv2.putText(
            marked,
            label,
            (bx1 + 6, by2 - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            thickness,
            cv2.LINE_AA,
        )
    return marked
