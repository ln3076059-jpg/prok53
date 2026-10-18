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
    positives = negatives = boxes = 0
    for record in records:
        record.validate()
        if record.image_sha256 in hashes:
            raise ValueError(f"duplicate image hash: {record.image_sha256}")
        hashes.add(record.image_sha256)
        contexts[record.context] += 1
        boxes += len(record.boxes)
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
        "contexts": dict(sorted(contexts.items())),
        "occupant_roles": dict(sorted(roles.items())),
        "blockers": blockers,
    }
