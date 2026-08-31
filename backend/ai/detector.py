from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from threading import Lock

import numpy as np
import yaml


@dataclass(frozen=True)
class NormalizedDetection:
    class_id: int
    class_name: str
    confidence: float
    xyxy: tuple[float, float, float, float]
    track_id: int | None = None


class SafetyDetector:
    """Single process-wide YOLO instance. The model is never reloaded per frame."""

    def __init__(self, weights: Path, config_path: Path):
        self.weights = weights
        self.config = yaml.safe_load(config_path.read_text())
        self.names = {int(key): value for key, value in self.config["class_names"].items()}
        self.thresholds = self.config["thresholds"]
        self.model_version = self.config.get("model_version", "UNKNOWN")
        self._model = None
        self._lock = Lock()

    @property
    def available(self) -> bool:
        return self.weights.exists()

    def load(self) -> None:
        if not self.available:
            raise FileNotFoundError(f"locked model weights not installed: {self.weights}")
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from ultralytics import YOLO

                    self._model = YOLO(str(self.weights))

    def predict(self, frame: np.ndarray, track: bool = True) -> list[NormalizedDetection]:
        self.load()
        minimum = min(self.thresholds.values())
        if track:
            results = self._model.track(frame, persist=True, tracker=self.config.get("tracker", "bytetrack.yaml"), conf=minimum, verbose=False)
        else:
            results = self._model.predict(frame, conf=minimum, verbose=False)
        return self.normalize(results[0])

    def normalize(self, result) -> list[NormalizedDetection]:
        output: list[NormalizedDetection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return output
        ids = boxes.id.cpu().tolist() if getattr(boxes, "id", None) is not None else [None] * len(boxes)
        for raw_box, raw_conf, raw_class, track_id in zip(boxes.xyxy.cpu().tolist(), boxes.conf.cpu().tolist(), boxes.cls.cpu().tolist(), ids, strict=True):
            class_id = int(raw_class)
            class_name = self.names.get(class_id)
            if class_name is None or float(raw_conf) < float(self.thresholds[class_name]):
                continue
            output.append(NormalizedDetection(class_id, class_name, float(raw_conf), tuple(float(value) for value in raw_box), int(track_id) if track_id is not None else None))
        return output

    def status(self) -> dict:
        return {"available": self.available, "loaded": self._model is not None, "model_version": self.model_version, "weights": str(self.weights), "threshold_status": self.config.get("threshold_status")}

