"""Autonomous source-frame phone proposals for the research candidate path."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from training.dmd.behavior_roi import Box, RoiStrategy
from training.dmd.high_res_roi import HighResolutionRoiPipeline, frame_sha256

FORBIDDEN_METADATA = {"ground_truth", "reviewed_box", "selected_box", "phone_box", "event_bounds"}


@dataclass(frozen=True)
class PhoneProposal:
    box: Box
    confidence: float
    source: str
    model_sha256: str


@dataclass(frozen=True)
class DriverRegion:
    box: Box | None
    track_id: str
    role: str
    confidence: float
    verified: bool
    source: str


@dataclass(frozen=True)
class PhoneContextEvidence:
    state: str = "UNKNOWN"
    verified: bool = False
    source: str = "NONE"

    def __post_init__(self) -> None:
        if self.state not in {"UNKNOWN", "HANDHELD", "MOUNTED_PHONE", "MOUNTED_CAMERA"}:
            raise ValueError("unsupported phone context")
        if self.state == "UNKNOWN" and self.verified:
            raise ValueError("UNKNOWN context cannot be verified")


class PhoneDetector(Protocol):
    model_sha256: str

    def detect(self, frame: np.ndarray, confidence_floor: float) -> list[PhoneProposal]: ...


def local_to_original(box: Box, origin: tuple[float, float]) -> Box:
    return Box(box.x1 + origin[0], box.y1 + origin[1], box.x2 + origin[0], box.y2 + origin[1])


def original_to_local(box: Box, origin: tuple[float, float]) -> Box:
    return Box(box.x1 - origin[0], box.y1 - origin[1], box.x2 - origin[0], box.y2 - origin[1])


class YoloPhoneDetector:
    """Lazy, single-class YOLO adapter; loading weights never trains or activates them."""

    def __init__(self, weights: Path, *, image_size: int = 640):
        if not weights.is_file():
            raise FileNotFoundError(weights)
        self.weights = weights
        self.image_size = image_size
        self.model_sha256 = hashlib.sha256(weights.read_bytes()).hexdigest()
        self._model = None

    def _load(self):
        if self._model is None:
            from ultralytics import YOLO

            self._model = YOLO(str(self.weights))
            if list(self._model.names.values()) != ["phone"]:
                raise ValueError("phone detector must have exactly one class named phone")

    def detect(self, frame: np.ndarray, confidence_floor: float) -> list[PhoneProposal]:
        if not 0 <= confidence_floor <= 1:
            raise ValueError("confidence floor must be in [0, 1]")
        self._load()
        result = self._model.predict(
            frame, imgsz=self.image_size, conf=confidence_floor, iou=0.7,
            max_det=20, device="cpu", verbose=False,
        )[0]
        return [
            PhoneProposal(Box(*map(float, xyxy)), float(score), "YOLO_PHONE", self.model_sha256)
            for xyxy, score in zip(
                result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(), strict=True
            )
        ]


class AutonomousPhonePipeline:
    def __init__(self, detector: PhoneDetector, confidence_floor: float = 0.02):
        if not 0 <= confidence_floor <= 1:
            raise ValueError("confidence floor must be in [0, 1]")
        self.detector = detector
        self.confidence_floor = confidence_floor
        self.roi = HighResolutionRoiPipeline()
        self.config_sha256 = hashlib.sha256(
            json.dumps(
                {
                    "confidence_floor": confidence_floor,
                    "detector_sha256": detector.model_sha256,
                    "roi": "HIGH_RES_DYNAMIC_PADDED_V1",
                },
                sort_keys=True,
            ).encode()
        ).hexdigest()

    def process(
        self,
        frame: np.ndarray,
        metadata: dict,
        driver: DriverRegion,
        *,
        hand_boxes: tuple[Box, ...] = (),
        face_box: Box | None = None,
        context: PhoneContextEvidence | None = None,
    ) -> dict:
        forbidden = FORBIDDEN_METADATA.intersection(metadata)
        if forbidden:
            raise ValueError(f"autonomous mode rejects oracle metadata: {sorted(forbidden)}")
        if frame.ndim != 3 or frame.shape[2] != 3:
            raise ValueError("expected HxWx3 source frame")
        height, width = frame.shape[:2]
        context = context or PhoneContextEvidence()
        raw_proposals = self.detector.detect(frame, self.confidence_floor)
        proposals, rejected = [], []
        for proposal in raw_proposals:
            if proposal.confidence < self.confidence_floor:
                rejected.append(
                    {"proposal": asdict(proposal), "reason": "BELOW_CONFIDENCE_FLOOR"}
                )
            else:
                proposals.append(proposal)
        for proposal in proposals:
            proposal.box.clamp(width, height)
            if not np.isfinite(proposal.confidence) or not 0 <= proposal.confidence <= 1:
                raise ValueError("invalid detector confidence")
            if proposal.model_sha256 != self.detector.model_sha256:
                raise ValueError("proposal model hash mismatch")
        driver_box = driver.box or Box(0, 0, width, height)
        outputs = []
        if proposals:
            for proposal in proposals:
                crop = self.roi.extract(
                    frame, driver_box, RoiStrategy.DYNAMIC_PADDED,
                    source_camera=str(metadata.get("camera", "UNKNOWN")),
                    phone_box=proposal.box, hand_boxes=hand_boxes, face_box=face_box,
                    driver_track_id=driver.track_id, confidence=proposal.confidence,
                )[0]
                mounted = context.verified and context.state in {
                    "MOUNTED_PHONE",
                    "MOUNTED_CAMERA",
                }
                reasons = []
                if mounted:
                    reasons.append("VERIFIED_MOUNTED_CONTEXT")
                if not driver.verified or driver.role != "driver":
                    reasons.append("NO_VERIFIED_DRIVER_ROLE")
                if not (context.verified and context.state == "HANDHELD"):
                    reasons.append("NO_VERIFIED_HANDHELD_CONTEXT")
                outputs.append(
                    {
                        "proposal": asdict(proposal),
                        "crop": crop.record(),
                        "observation": {
                            "detector_confidence": proposal.confidence,
                            "phone_context": asdict(context),
                            "driver_track_id": driver.track_id,
                            "driver_role": driver.role if driver.verified else "UNKNOWN",
                            "admissible_for_driver_fusion": not reasons,
                            "reason_codes": reasons,
                        },
                    }
                )
            status = "PHONE_PROPOSALS"
        else:
            strategy = RoiStrategy.FACE_HAND if hand_boxes or face_box else RoiStrategy.UPPER_BODY
            crop = self.roi.extract(
                frame, driver_box, strategy,
                source_camera=str(metadata.get("camera", "UNKNOWN")),
                hand_boxes=hand_boxes, face_box=face_box,
                driver_track_id=driver.track_id,
            )[0]
            outputs.append({"proposal": None, "crop": crop.record()})
            status = "NO_PHONE_PROPOSAL"
        return {
            "mode": "AUTONOMOUS_DETECTION_EVALUATION",
            "frame_sha256": frame_sha256(frame),
            "metadata": dict(metadata),
            "driver": asdict(driver),
            "driver_role_state": "VERIFIED" if driver.verified else "UNKNOWN",
            "status": status,
            "outputs": outputs,
            "rejected_proposals": rejected,
            "model_sha256": self.detector.model_sha256,
            "config_sha256": self.config_sha256,
            "phone_ap": None,
            "event_recovery_claimed": False,
        }
