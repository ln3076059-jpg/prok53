from __future__ import annotations

from dataclasses import dataclass, replace
from math import hypot

from backend.ai.detector import NormalizedDetection


def box_iou(a: tuple[float, float, float, float], b: tuple[float, float, float, float]) -> float:
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    union = area_a + area_b - intersection
    return intersection / union if union > 0 else 0.0


@dataclass
class _Track:
    track_id: int
    class_name: str
    xyxy: tuple[float, float, float, float]
    last_frame: int


class WithinVehicleTracker:
    """Track-by-detection scoped to one vehicle context.

    Matching uses class, IoU, normalized center distance, and temporal continuity. It never
    treats two observations as identical merely because they share a class.
    """

    def __init__(
        self,
        max_gap_frames: int = 12,
        minimum_iou: float = 0.10,
        maximum_center_distance: float = 0.25,
    ):
        self.max_gap_frames = int(max_gap_frames)
        self.minimum_iou = float(minimum_iou)
        self.maximum_center_distance = float(maximum_center_distance)
        self._tracks: dict[str, dict[int, _Track]] = {}
        self._next_id = 1

    def update(
        self,
        vehicle_context_id: str,
        frame_number: int,
        detections: list[NormalizedDetection],
        frame_width: int,
        frame_height: int,
    ) -> list[NormalizedDetection]:
        tracks = self._tracks.setdefault(vehicle_context_id, {})
        expired = [
            track_id
            for track_id, track in tracks.items()
            if frame_number - track.last_frame > self.max_gap_frames
        ]
        for track_id in expired:
            del tracks[track_id]

        used: set[int] = set()
        output: list[NormalizedDetection] = []
        diagonal = max(hypot(frame_width, frame_height), 1.0)
        for detection in sorted(detections, key=lambda item: item.confidence, reverse=True):
            if detection.track_id is not None:
                track_id = int(detection.track_id)
            else:
                candidates: list[tuple[float, int]] = []
                dcx = (detection.xyxy[0] + detection.xyxy[2]) / 2
                dcy = (detection.xyxy[1] + detection.xyxy[3]) / 2
                for candidate_id, track in tracks.items():
                    if candidate_id in used or track.class_name != detection.class_name:
                        continue
                    tcx = (track.xyxy[0] + track.xyxy[2]) / 2
                    tcy = (track.xyxy[1] + track.xyxy[3]) / 2
                    distance = hypot(dcx - tcx, dcy - tcy) / diagonal
                    iou = box_iou(detection.xyxy, track.xyxy)
                    if iou < self.minimum_iou and distance > self.maximum_center_distance:
                        continue
                    recency = 1.0 - min(
                        1.0, (frame_number - track.last_frame) / max(self.max_gap_frames, 1)
                    )
                    candidates.append(
                        (0.65 * iou + 0.25 * (1.0 - distance) + 0.10 * recency, candidate_id)
                    )
                if candidates:
                    _, track_id = max(candidates)
                else:
                    track_id = self._next_id
                    self._next_id += 1
            used.add(track_id)
            tracks[track_id] = _Track(track_id, detection.class_name, detection.xyxy, frame_number)
            output.append(replace(detection, track_id=track_id))
        return output

    def reset_vehicle(self, vehicle_context_id: str) -> None:
        self._tracks.pop(vehicle_context_id, None)

    def reset(self) -> None:
        self._tracks.clear()
