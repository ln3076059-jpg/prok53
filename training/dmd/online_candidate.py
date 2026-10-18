"""Research-only E3 streaming features and a portable numeric linear head.

This module has no production registration and never fits a model. State is
bounded to one active stream; switching session, camera, or track resets it.
"""

from __future__ import annotations

import hashlib
import json
from collections import deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.causal_motion import (
    CausalMotionEncoder,
    spatial_residual_standardize,
)
from training.dmd.driver_phone_association import AssociationResult, AssociationState

CLASSES = ("SAFE_DRIVING", "PHONE_USE", "OTHER_ACTIVITY")


@dataclass(frozen=True)
class OnlineConfig:
    history: int = 120
    minimum_history: int = 8
    epsilon: float = 1e-3
    sample_period: float = 15 / 29.76
    cadence_tolerance: float = 0.02

    def __post_init__(self) -> None:
        if not 1 <= self.minimum_history <= self.history <= 10000:
            raise ValueError("invalid history limits")
        for value in (self.epsilon, self.sample_period, self.cadence_tolerance):
            if not np.isfinite(value) or value <= 0:
                raise ValueError("config values must be finite and positive")


@dataclass(frozen=True)
class FeatureResult:
    feature: np.ndarray | None
    ready: bool
    reasons: tuple[str, ...]
    history_count: int


class OnlineE3:
    def __init__(self, config: OnlineConfig | None = None):
        self.config = config or OnlineConfig()
        self.encoder = CausalMotionEncoder()
        self.frames: deque[np.ndarray] = deque(maxlen=8)
        self.history: deque[np.ndarray] = deque(maxlen=self.config.history)
        self.key: tuple[str, str, str] | None = None
        self.timestamp: float | None = None

    def reset(self) -> None:
        self.frames.clear()
        self.history.clear()
        self.timestamp = None
        self.key = None

    def push(
        self, frame: np.ndarray | None, timestamp: float, key: tuple[str, str, str]
    ) -> FeatureResult:
        if len(key) != 3 or any(not isinstance(v, str) or not v for v in key):
            raise ValueError("key requires session, camera, track (or UNKNOWN)")
        if not np.isfinite(timestamp):
            raise ValueError("timestamp must be finite")
        if key == self.key and self.timestamp is not None and timestamp <= self.timestamp:
            raise ValueError("timestamps must strictly increase within a stream")
        if frame is not None and (frame.dtype != np.uint8 or frame.shape != (64, 64, 3)):
            raise ValueError("E3 requires an RGB uint8 64x64 frame with recorded preprocessing")
        reasons: list[str] = []
        if self.key != key:
            self.reset()
            reasons.append("STREAM_RESET")
        elif self.timestamp is not None:
            gap = timestamp - self.timestamp
            if abs(gap - self.config.sample_period) > self.config.cadence_tolerance:
                self.reset()
                reasons.append("CADENCE_GAP_RESET")
        if frame is None:
            self.reset()
            self.key, self.timestamp = key, timestamp
            return FeatureResult(None, False, tuple(reasons + ["MISSING_VIEW"]), 0)
        self.key, self.timestamp = key, timestamp
        self.frames.append(frame.copy())
        raw = self.encoder.encode_at(np.stack(self.frames), len(self.frames) - 1)
        spatial = spatial_residual_standardize(raw[None, :])[0].astype(np.float64)
        count = len(self.history)
        ready = count >= self.config.minimum_history
        feature = np.zeros_like(spatial, dtype=np.float32)
        if ready:
            previous = np.stack(self.history)
            feature = (
                (spatial - previous.mean(axis=0)) / (previous.std(axis=0) + self.config.epsilon)
            ).astype(np.float32)
        else:
            reasons.append("INSUFFICIENT_HISTORY")
        self.history.append(spatial)
        return FeatureResult(feature, ready, tuple(reasons), count)


class NumericHead:
    """JSON-only StandardScaler + three-class OvR logistic inference.

    Scores are normalized OvR sigmoid outputs, explicitly uncalibrated.
    Export requires supplied provenance and already-fitted sklearn objects.
    """

    def __init__(self, payload: dict[str, Any]):
        schema = payload.get("schema")
        if schema not in {1, 2} or tuple(payload.get("classes", [])) != CLASSES:
            raise ValueError("unsupported head schema/class mapping")
        self.mean = np.asarray(payload["mean"], dtype=np.float64)
        self.scale = np.asarray(payload["scale"], dtype=np.float64)
        self.weights = np.asarray(payload["weights"], dtype=np.float64)
        self.bias = np.asarray(payload["bias"], dtype=np.float64)
        if (
            self.mean.ndim != 1
            or self.scale.shape != self.mean.shape
            or self.weights.shape != (3, len(self.mean))
            or self.bias.shape != (3,)
        ):
            raise ValueError("invalid head dimensions")
        if any(not np.isfinite(a).all() for a in (self.mean, self.scale, self.weights, self.bias)):
            raise ValueError("nonfinite head parameters")
        if np.any(self.scale <= 0):
            raise ValueError("scale must be positive")
        provenance = payload.get("provenance", {})
        for key in ("dataset_hash", "split_hash", "feature_version", "seed", "training_subjects"):
            if key not in provenance:
                raise ValueError(f"missing model provenance: {key}")
        if schema == 2:
            for key in ("checkpoint_hash", "config_hash", "library_versions"):
                if key not in provenance:
                    raise ValueError(f"missing model provenance: {key}")
            if payload.get("calibration") not in {
                "UNCALIBRATED_OVR_SIGMOID",
                "CALIBRATED_ON_LOCKED_VALIDATION",
            }:
                raise ValueError("invalid calibration status")
        self.payload = payload
        self.sha256 = hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    def predict(self, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x = np.asarray(features, dtype=np.float64)
        if x.ndim != 2 or x.shape[1] != len(self.mean) or not np.isfinite(x).all():
            raise ValueError("invalid feature matrix")
        decision = ((x - self.mean) / self.scale) @ self.weights.T + self.bias
        # log(sigmoid) followed by normalization avoids all-zero underflow.
        log_prob = -np.logaddexp(0.0, -decision)
        probs = np.exp(log_prob - log_prob.max(axis=1, keepdims=True))
        probs /= probs.sum(axis=1, keepdims=True)
        return decision.argmax(axis=1), probs

    def save(self, path: Path) -> None:
        path.write_text(
            json.dumps(self.payload, indent=2, allow_nan=False) + "\n", encoding="utf-8"
        )

    @classmethod
    def load(cls, path: Path, expected_sha256: str | None = None) -> NumericHead:
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if expected_sha256 is not None and actual != expected_sha256:
            raise ValueError("model artifact hash mismatch")
        return cls(json.loads(payload.decode("utf-8")))

    @classmethod
    def from_fitted(cls, scaler: Any, classifier: Any, provenance: dict) -> NumericHead:
        if not np.array_equal(classifier.classes_, np.arange(3)):
            raise ValueError("expected fitted classes 0, 1, 2")
        return cls(
            {
                "schema": 1,
                "classes": list(CLASSES),
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
                "weights": [e.coef_[0].tolist() for e in classifier.estimators_],
                "bias": [float(e.intercept_[0]) for e in classifier.estimators_],
                "provenance": provenance,
                "calibration": "UNCALIBRATED_OVR_SIGMOID",
            }
        )


class ResearchCandidate:
    """Opt-in adapter to the existing event engine; no alternate event policy."""

    def __init__(self, head: NumericHead, temporal_config: dict, *, enabled: bool = False):
        self.head, self.enabled = head, enabled
        self.stream = OnlineE3()
        self.temporal_config = dict(temporal_config)
        self.engine = TemporalEventEngine(**self.temporal_config)
        self.config_sha256 = hashlib.sha256(
            json.dumps(self.temporal_config, sort_keys=True).encode()
        ).hexdigest()

    def push(
        self,
        frame: np.ndarray | None,
        timestamp: float,
        key: tuple[str, str, str],
        association: AssociationResult,
        raw_phone_score: float,
        phone_context: str,
    ) -> dict[str, Any]:
        if not self.enabled:
            raise RuntimeError("research candidate is disabled")
        if not np.isfinite(raw_phone_score) or not 0 <= raw_phone_score <= 1:
            raise ValueError("invalid raw phone score")
        result = self.stream.push(frame, timestamp, key)
        if any(r.endswith("RESET") for r in result.reasons):
            self.engine = TemporalEventEngine(**self.temporal_config)
        scores = None
        if result.feature is not None:
            _, probabilities = self.head.predict(result.feature[None, :])
            scores = dict(zip(CLASSES, probabilities[0].tolist(), strict=True))
        reasons = list(result.reasons)
        admitted = (
            result.ready
            and association.state == AssociationState.DRIVER_ASSOCIATED.value
            and phone_context == "HANDHELD_USE"
            and raw_phone_score > 0
        )
        if not admitted:
            reasons.append("NEEDS_REVIEW_OR_NO_VERIFIED_DRIVER_EVIDENCE")
            # Unknown evidence cannot bridge a prior verified event in this candidate.
            self.engine = TemporalEventEngine(**self.temporal_config)
        fused = min(raw_phone_score, scores["PHONE_USE"]) if admitted and scores else None
        events = []
        if fused is not None:
            obs = Observation(
                timestamp=timestamp,
                class_name="phone",
                confidence=raw_phone_score,
                track_id=None,
                occupant_role="driver",
                vehicle_context_id=key[0],
                occupant_id=association.occupant_track_id,
                phone_context=phone_context,
                fusion_score=fused,
                evidence_source="V12_RESEARCH_CANDIDATE",
            )
            for ev in self.engine.add(obs):
                record = asdict(ev)
                record.update(
                    emission_time=timestamp,
                    source_key=list(key),
                    model_sha256=self.head.sha256,
                    config_sha256=self.config_sha256,
                )
                record["event_id"] = hashlib.sha256(
                    json.dumps(record, sort_keys=True).encode()
                ).hexdigest()[:24]
                events.append(record)
        return {
            "timestamp": timestamp,
            "source_key": list(key),
            "ready": result.ready,
            "action_scores": scores,
            "calibration": "UNCALIBRATED",
            "raw_phone_score": raw_phone_score,
            "association": association.as_dict(),
            "fused_score": fused,
            "fusion_rule": "MIN_OBJECT_ACTION_RESEARCH_ONLY",
            "abstained": not admitted,
            "reasons": reasons,
            "events": events,
            "model_sha256": self.head.sha256,
            "config_sha256": self.config_sha256,
        }
