from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

FEATURE_NAMES = (
    "detector_confidence",
    "classifier_unfastened",
    "classifier_fastened",
    "phone_hand_proximity",
    "phone_face_proximity",
    "pose_confidence",
    "temporal_ratio",
    "driver_role",
    "vehicle_context",
)


@dataclass(frozen=True)
class FusionFeatures:
    detector_confidence: float = 0.0
    classifier_unfastened: float = 0.0
    classifier_fastened: float = 0.0
    phone_hand_proximity: float = 0.0
    phone_face_proximity: float = 0.0
    pose_confidence: float = 0.0
    temporal_ratio: float = 0.0
    driver_role: float = 0.0
    vehicle_context: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {name: float(getattr(self, name)) for name in FEATURE_NAMES}


@dataclass(frozen=True)
class FusionResult:
    score: float
    decision: bool
    source: str


class LogisticFusion:
    """Small, dependency-free inference wrapper for an offline-trained logistic model."""

    def __init__(self, artifact: Path | None, threshold: float = 0.5):
        self.artifact = artifact
        self.threshold = float(threshold)
        self.model: dict | None = None
        if artifact and artifact.is_file():
            payload = json.loads(artifact.read_text(encoding="utf-8"))
            if payload.get("feature_names") != list(FEATURE_NAMES):
                raise ValueError("fusion artifact feature schema mismatch")
            self.model = payload
            self.threshold = float(payload.get("threshold", threshold))

    @property
    def available(self) -> bool:
        return self.model is not None

    def predict(self, features: FusionFeatures) -> FusionResult:
        values = features.as_dict()
        if not self.model:
            # Conservative fallback: context is mandatory and contradictions reduce confidence.
            score = values["detector_confidence"] * values["vehicle_context"]
            if values["classifier_unfastened"] or values["classifier_fastened"]:
                agreement = values["classifier_unfastened"]
                contradiction = values["classifier_fastened"]
                score *= max(0.0, min(1.0, 0.5 + 0.5 * agreement - 0.5 * contradiction))
            return FusionResult(score, score >= self.threshold, "CONSERVATIVE_RULE_FALLBACK")

        means = self.model["means"]
        scales = self.model["scales"]
        weights = self.model["weights"]
        logit = float(self.model["intercept"])
        for name in FEATURE_NAMES:
            scale = max(float(scales[name]), 1e-12)
            standardized = (values[name] - float(means[name])) / scale
            logit += standardized * float(weights[name])
        logit = max(-60.0, min(60.0, logit))
        score = 1.0 / (1.0 + math.exp(-logit))
        return FusionResult(score, score >= self.threshold, "LOGISTIC_FUSION")

    def status(self) -> dict:
        return {
            "available": self.available,
            "artifact": str(self.artifact) if self.artifact else None,
            "threshold": self.threshold,
            "mode": "LOGISTIC_FUSION" if self.available else "CONSERVATIVE_RULE_FALLBACK",
        }
