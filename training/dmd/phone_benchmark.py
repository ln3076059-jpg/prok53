"""Fail-closed schema and validator for a scientific phone detection benchmark."""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from math import isfinite

PHONE_CONTEXTS = {
    "HANDHELD",
    "NEAR_EAR",
    "IN_LAP",
    "MOUNTED_PHONE",
    "MOUNTED_CAMERA",
    "PASSENGER_PHONE",
    "PHONE_LIKE_OBJECT",
    "OTHER_NEGATIVE",
}
VISIBILITIES = {"VISIBLE", "PARTIALLY_OCCLUDED"}
OCCUPANT_ROLES = {"DRIVER", "PASSENGER", "UNKNOWN"}


@dataclass(frozen=True)
class PhoneBox:
    xyxy: tuple[float, float, float, float]
    visibility: str
    occupant_role: str
    association_confidence: float | None = None

    @property
    def area(self) -> float:
        x1, y1, x2, y2 = self.xyxy
        return max(0.0, x2 - x1) * max(0.0, y2 - y1)

    @property
    def is_small(self) -> bool:
        # Standard COCO definition for small objects: area < 32^2 = 1024 px^2
        return self.area < 1024.0

    def validate(self, width: int, height: int) -> None:
        x1, y1, x2, y2 = self.xyxy
        if not all(isfinite(value) for value in self.xyxy):
            raise ValueError("phone box coordinates must be finite")
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("phone box must have positive area inside the original image")
        if self.visibility not in VISIBILITIES:
            raise ValueError("invalid phone visibility")
        if self.occupant_role not in OCCUPANT_ROLES:
            raise ValueError("invalid occupant role")
        if self.association_confidence is not None and not (
            isfinite(self.association_confidence) and 0 <= self.association_confidence <= 1
        ):
            raise ValueError("association confidence must be in [0, 1]")


@dataclass(frozen=True)
class PhoneBenchmarkImage:
    image_sha256: str
    source_id: str
    source_asset_id: str
    source_license: str
    width: int
    height: int
    context: str
    reviewer_id: str
    review_status: str
    annotation_scope: str
    negative_scope_reviewed: bool
    boxes: tuple[PhoneBox, ...] = field(default_factory=tuple)
    subject_id: str | None = None
    session_id: str | None = None
    original_asset_id: str | None = None

    def validate(self) -> None:
        if len(self.image_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.image_sha256.lower()
        ):
            raise ValueError("image_sha256 must be a 64-character hex digest")
        if not self.source_id or not self.source_asset_id or not self.source_license:
            raise ValueError("source provenance and license are required")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("original image dimensions must be positive")
        if self.context not in PHONE_CONTEXTS:
            raise ValueError("invalid benchmark context")
        if self.review_status != "APPROVED" or not self.reviewer_id:
            raise ValueError("benchmark records require human approval")
        if self.annotation_scope != "ALL_VISIBLE_PHONES":
            raise ValueError("annotation scope must cover every visible phone")
        for box in self.boxes:
            box.validate(self.width, self.height)
        if not self.boxes and not self.negative_scope_reviewed:
            raise ValueError("zero-box images require explicit negative-scope review")
        if self.boxes and self.negative_scope_reviewed:
            raise ValueError("positive images cannot be marked as reviewed negatives")


def validate_phone_benchmark(records: list[PhoneBenchmarkImage]) -> dict:
    """Validate benchmark truth without treating missing boxes as negatives."""
    if not records:
        raise ValueError("phone benchmark is empty")
    hashes: set[str] = set()
    contexts: Counter[str] = Counter()
    roles: Counter[str] = Counter()
    positives = negatives = boxes = small_boxes = 0
    for record in records:
        record.validate()
        if record.image_sha256 in hashes:
            raise ValueError(f"duplicate image hash: {record.image_sha256}")
        hashes.add(record.image_sha256)
        contexts[record.context] += 1
        boxes += len(record.boxes)
        small_boxes += sum(1 for box in record.boxes if box.is_small)
        if record.boxes:
            positives += 1
            roles.update(box.occupant_role for box in record.boxes)
        else:
            negatives += 1
    blockers = []
    if positives == 0:
        blockers.append("NO_POSITIVE_IMAGES")
    if negatives == 0:
        blockers.append("NO_REVIEWED_NEGATIVE_IMAGES")
    if roles.get("DRIVER", 0) == 0:
        blockers.append("NO_DRIVER_PHONE_BOXES")
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "images": len(records),
        "positive_images": positives,
        "reviewed_negative_images": negatives,
        "phone_boxes": boxes,
        "small_phone_boxes": small_boxes,
        "contexts": dict(sorted(contexts.items())),
        "occupant_roles": dict(sorted(roles.items())),
        "blockers": blockers,
    }


SEATBELT_BENCHMARK_CLASSES = {"FASTENED", "UNFASTENED", "UNCERTAIN_OR_OCCLUDED"}


@dataclass(frozen=True)
class SeatbeltBenchmarkRecord:
    sample_id: str
    image_sha256: str
    source_id: str
    source_license: str
    torso_xyxy: tuple[float, float, float, float]
    state: str
    reviewer_id: str
    review_status: str
    occlusion_reason: str | None = None
    subject_id: str | None = None

    def validate(self, width: int = 1920, height: int = 1080) -> None:
        if len(self.image_sha256) != 64 or any(
            char not in "0123456789abcdef" for char in self.image_sha256.lower()
        ):
            raise ValueError("image_sha256 must be a 64-character hex digest")
        if self.state not in SEATBELT_BENCHMARK_CLASSES:
            raise ValueError(f"invalid seatbelt benchmark state: {self.state}")
        if self.review_status != "APPROVED" or not self.reviewer_id:
            raise ValueError("seatbelt benchmark records require authenticated human approval")
        if self.state == "UNCERTAIN_OR_OCCLUDED" and not self.occlusion_reason:
            raise ValueError("UNCERTAIN_OR_OCCLUDED requires explicit occlusion_reason")
        x1, y1, x2, y2 = self.torso_xyxy
        if not (0 <= x1 < x2 <= width and 0 <= y1 < y2 <= height):
            raise ValueError("torso coordinates must have positive area within bounds")


def validate_seatbelt_benchmark(records: list[SeatbeltBenchmarkRecord]) -> dict:
    if not records:
        raise ValueError("seatbelt benchmark is empty")
    seen_ids: set[str] = set()
    states: Counter[str] = Counter()
    for record in records:
        record.validate()
        if record.sample_id in seen_ids:
            raise ValueError(f"duplicate seatbelt sample_id: {record.sample_id}")
        seen_ids.add(record.sample_id)
        states[record.state] += 1
    blockers = []
    for required in ("FASTENED", "UNFASTENED", "UNCERTAIN_OR_OCCLUDED"):
        if states.get(required, 0) == 0:
            blockers.append(f"MISSING_{required}_SAMPLES")
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "total_records": len(records),
        "state_counts": dict(sorted(states.items())),
        "blockers": blockers,
    }


PHONE_EVENT_BENCHMARK_TYPES = {"SAFE_DRIVING", "PHONE_USE", "OTHER_ACTIVITY", "UNKNOWN"}


@dataclass(frozen=True)
class PhoneEventBenchmarkRecord:
    event_id: str
    video_id: str
    source_id: str
    source_license: str
    start_timestamp: float
    end_timestamp: float
    event_type: str
    camera_id: str
    reviewer_id: str
    review_status: str
    subject_id: str | None = None
    is_static_image_synthetic: bool = False

    def validate(self) -> None:
        if self.is_static_image_synthetic:
            raise ValueError("synthetic timestamps from static images are strictly forbidden")
        if not self.video_id or not self.source_id:
            raise ValueError("video_id and source_id are required")
        if not (isfinite(self.start_timestamp) and isfinite(self.end_timestamp)):
            raise ValueError("timestamps must be finite numbers")
        if self.end_timestamp <= self.start_timestamp:
            raise ValueError("end_timestamp must be strictly greater than start_timestamp")
        if self.event_type not in PHONE_EVENT_BENCHMARK_TYPES:
            raise ValueError(f"invalid event type: {self.event_type}")
        if self.review_status != "APPROVED" or not self.reviewer_id:
            raise ValueError("event records require authenticated human approval")


def validate_phone_event_benchmark(records: list[PhoneEventBenchmarkRecord]) -> dict:
    if not records:
        raise ValueError("phone event benchmark is empty")
    seen_ids: set[str] = set()
    event_counts: Counter[str] = Counter()
    total_duration = 0.0
    for record in records:
        record.validate()
        if record.event_id in seen_ids:
            raise ValueError(f"duplicate event_id: {record.event_id}")
        seen_ids.add(record.event_id)
        event_counts[record.event_type] += 1
        total_duration += record.end_timestamp - record.start_timestamp
    blockers = []
    if event_counts.get("PHONE_USE", 0) == 0:
        blockers.append("NO_PHONE_USE_EVENTS")
    if event_counts.get("SAFE_DRIVING", 0) == 0:
        blockers.append("NO_SAFE_DRIVING_EVENTS")
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "total_events": len(records),
        "total_duration_seconds": total_duration,
        "event_counts": dict(sorted(event_counts.items())),
        "blockers": blockers,
    }


@dataclass(frozen=True)
class DriverAssociationBenchmarkRecord:
    sample_id: str
    image_sha256: str
    occupant_track_id: str
    true_role: str
    phone_box_xyxy: tuple[float, float, float, float]
    occupant_box_xyxy: tuple[float, float, float, float]
    association_confidence: float
    reviewer_id: str
    review_status: str
    camera_view: str = "BODY"

    def validate(self) -> None:
        if self.true_role not in OCCUPANT_ROLES:
            raise ValueError(f"invalid occupant role: {self.true_role}")
        if self.review_status != "APPROVED" or not self.reviewer_id:
            raise ValueError("driver association benchmark requires human approval")
        if not (isfinite(self.association_confidence) and 0 <= self.association_confidence <= 1):
            raise ValueError("association_confidence must be in [0, 1]")


def evaluate_phone_benchmark(
    records: list[PhoneBenchmarkImage],
    detector_proposals: dict[str, list[tuple[float, float, float, float, float]]] | None = None,
) -> dict:
    validation = validate_phone_benchmark(records)
    if validation["status"] != "READY":
        return {
            "status": "BLOCKED",
            "ap_metrics_allowed": False,
            "blockers": validation["blockers"],
            "d0_d4_metrics": None,
        }
    return {
        "status": "READY",
        "ap_metrics_allowed": True,
        "validation": validation,
        "d0_d4_metrics": {
            "ap50": None,
            "ap50_95": None,
            "small_phone_recall": None,
            "negative_frame_fp_rate": None,
        },
    }
