from __future__ import annotations

import hashlib
from dataclasses import dataclass
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
        self.config_path = config_path
        if not config_path.is_file():
            raise FileNotFoundError(f"model config not found: {config_path}")
        self._config_bytes = config_path.read_bytes()
        self.config_sha256 = hashlib.sha256(self._config_bytes).hexdigest()
        self.config = yaml.safe_load(self._config_bytes.decode("utf-8"))
        if not isinstance(self.config, dict):
            raise ValueError("model config must be a YAML mapping")
        for required in ("class_names", "thresholds", "vehicle_context", "temporal"):
            if required not in self.config:
                raise ValueError(f"model config missing required section: {required}")
        self.names = {int(key): value for key, value in self.config["class_names"].items()}
        self.thresholds = self.config["thresholds"]
        self.model_version = self.config.get("model_version", "UNKNOWN")
        self._model = None
        self.specialists = self.config.get("specialized_detectors", {})
        self._specialist_models: dict[str, object] = {}
        self._lock = Lock()

    @staticmethod
    def _sha256(path: Path) -> str | None:
        if not path.is_file():
            return None
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()

    @property
    def activation_state(self) -> str:
        serialized = yaml.safe_dump(self.config).upper()
        if "UNTRAINED" in serialized:
            return "UNTRAINED"
        if not self.available:
            return "ARTIFACTS_MISSING"
        if any(marker in serialized for marker in ("PLACEHOLDER", "PENDING", "NOT_APPROVED")):
            return "NOT_CALIBRATED"
        return str(self.config.get("activation_state", "CANDIDATE_NOT_PRODUCTION_APPROVED"))

    @property
    def available(self) -> bool:
        if self.specialists.get("enabled"):
            entries = self.specialists.get("models", [])
            return bool(entries) and all(Path(item["weights"]).is_file() for item in entries)
        return self.weights.exists()

    def load(self) -> None:
        if not self.available:
            raise FileNotFoundError(f"locked model weights not installed: {self.weights}")
        if self.specialists.get("enabled"):
            with self._lock:
                if not self._specialist_models:
                    from ultralytics import YOLO

                    self._specialist_models = {
                        item["name"]: YOLO(str(Path(item["weights"])))
                        for item in self.specialists["models"]
                    }
            return
        if self._model is None:
            with self._lock:
                if self._model is None:
                    from ultralytics import YOLO

                    self._model = YOLO(str(self.weights))

    def predict(self, frame: np.ndarray, track: bool = True) -> list[NormalizedDetection]:
        self.load()
        if self.specialists.get("enabled"):
            return self._predict_specialists(frame, track)
        minimum = min(self.thresholds.values())
        if track:
            results = self._model.track(
                frame,
                persist=True,
                tracker=self.config.get("tracker", "bytetrack.yaml"),
                conf=minimum,
                verbose=False,
            )
        else:
            results = self._model.predict(frame, conf=minimum, verbose=False)
        return self.normalize(results[0])

    def _predict_specialists(self, frame: np.ndarray, track: bool) -> list[NormalizedDetection]:
        output: list[NormalizedDetection] = []
        for model_index, item in enumerate(self.specialists["models"]):
            model = self._specialist_models[item["name"]]
            source_map = {int(key): value for key, value in item["class_map"].items()}
            mapped_thresholds = [
                float(self.thresholds[name])
                for name in source_map.values()
                if name in self.thresholds
            ]
            minimum = float(item.get("threshold", min(mapped_thresholds, default=0.25)))
            if track:
                result = model.track(frame, persist=True, conf=minimum, verbose=False)[0]
            else:
                result = model.predict(frame, conf=minimum, verbose=False)[0]
            boxes = getattr(result, "boxes", None)
            if boxes is None:
                continue
            ids = (
                boxes.id.cpu().tolist()
                if getattr(boxes, "id", None) is not None
                else [None] * len(boxes)
            )
            for raw_box, raw_conf, raw_class, raw_track_id in zip(
                boxes.xyxy.cpu().tolist(),
                boxes.conf.cpu().tolist(),
                boxes.cls.cpu().tolist(),
                ids,
                strict=True,
            ):
                class_name = source_map.get(int(raw_class))
                threshold = float(self.thresholds.get(class_name, item.get("threshold", 0.25)))
                if class_name is None or float(raw_conf) < threshold:
                    continue
                class_id = next((key for key, name in self.names.items() if name == class_name), -1)
                track_id = None
                if raw_track_id is not None:
                    track_id = model_index * 1_000_000 + int(raw_track_id)
                output.append(
                    NormalizedDetection(
                        class_id,
                        class_name,
                        float(raw_conf),
                        tuple(float(value) for value in raw_box),
                        track_id,
                    )
                )
        return output

    def normalize(self, result) -> list[NormalizedDetection]:
        output: list[NormalizedDetection] = []
        boxes = getattr(result, "boxes", None)
        if boxes is None:
            return output
        ids = (
            boxes.id.cpu().tolist()
            if getattr(boxes, "id", None) is not None
            else [None] * len(boxes)
        )
        for raw_box, raw_conf, raw_class, track_id in zip(
            boxes.xyxy.cpu().tolist(),
            boxes.conf.cpu().tolist(),
            boxes.cls.cpu().tolist(),
            ids,
            strict=True,
        ):
            class_id = int(raw_class)
            class_name = self.names.get(class_id)
            if class_name is None or float(raw_conf) < float(self.thresholds[class_name]):
                continue
            output.append(
                NormalizedDetection(
                    class_id,
                    class_name,
                    float(raw_conf),
                    tuple(float(value) for value in raw_box),
                    int(track_id) if track_id is not None else None,
                )
            )
        return output

    def status(self) -> dict:
        auxiliary = self.config.get("auxiliary_models", {})
        vehicle = self.config.get("vehicle_context", {})
        specialist_hashes = {
            item.get("name", "unknown"): self._sha256(Path(item["weights"]))
            for item in self.specialists.get("models", [])
            if item.get("weights")
        }
        return {
            "available": self.available,
            "loaded": self._model is not None or bool(self._specialist_models),
            "model_version": self.model_version,
            "weights": str(self.weights),
            "weights_sha256": self._sha256(self.weights),
            "specialist_weight_sha256": specialist_hashes,
            "config": str(self.config_path),
            "config_sha256": self.config_sha256,
            "activation_state": self.activation_state,
            "threshold_status": self.config.get("threshold_status"),
            "specialized_detectors": {
                "enabled": bool(self.specialists.get("enabled")),
                "models": [item.get("name") for item in self.specialists.get("models", [])],
            },
            "auxiliary_models": {
                "enabled": bool(auxiliary.get("enabled")),
                "seatbelt_classifier_available": bool(
                    auxiliary.get("seatbelt_classifier_weights")
                    and Path(auxiliary["seatbelt_classifier_weights"]).is_file()
                ),
                "pose_available": bool(
                    auxiliary.get("pose_weights") and Path(auxiliary["pose_weights"]).is_file()
                ),
                "fusion_available": bool(
                    auxiliary.get("fusion_artifact")
                    and Path(auxiliary["fusion_artifact"]).is_file()
                ),
                "vehicle_detector_available": bool(
                    vehicle.get("vehicle_weights") and Path(vehicle["vehicle_weights"]).is_file()
                ),
            },
        }
