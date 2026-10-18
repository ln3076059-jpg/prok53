"""Shared data-governance contracts for Roadwatch research datasets."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from enum import StrEnum


class DatasetStatus(StrEnum):
    DISCOVERED = "DISCOVERED"
    LICENSE_UNVERIFIED = "LICENSE_UNVERIFIED"
    ACCESS_REQUEST_REQUIRED = "ACCESS_REQUEST_REQUIRED"
    DOWNLOAD_AUTHORIZED = "DOWNLOAD_AUTHORIZED"
    DOWNLOADED = "DOWNLOADED"
    QUARANTINED_FOR_REVIEW = "QUARANTINED_FOR_REVIEW"
    ANNOTATION_PENDING = "ANNOTATION_PENDING"
    APPROVED_FOR_TRAINING = "APPROVED_FOR_TRAINING"
    VALIDATION_ONLY = "VALIDATION_ONLY"
    LOCKED_TEST = "LOCKED_TEST"
    BLOCKED = "BLOCKED"


PHONE_VISIBILITY = {
    "PHONE_BOX",
    "PHONE_VISIBLE",
    "PHONE_OCCLUDED",
    "PHONE_ABSENT",
    "UNKNOWN_PHONE_VISIBILITY",
}
PHONE_ACTION = {
    "SAFE_DRIVING",
    "TEXTING",
    "HANDHELD_CALL",
    "OTHER_PHONE_USE",
    "OTHER_ACTIVITY",
    "UNKNOWN_ACTION",
}
PHONE_CONTEXT = {
    "HANDHELD",
    "MOUNTED",
    "STATIC_CAMERA",
    "PASSENGER_PHONE",
    "UNKNOWN_CONTEXT",
}
OCCUPANT_ROLES = {"DRIVER", "FRONT_PASSENGER", "REAR_PASSENGER", "UNKNOWN"}
SEATBELT = {"FASTENED", "UNFASTENED", "UNCERTAIN_OR_OCCLUDED"}
PROTECTED_SUBJECTS = frozenset({"gZ-37", "gE-28"})


def require_download_authorized(source: dict) -> None:
    """Allow network acquisition only after an explicit registry authorization."""
    if source.get("status") != DatasetStatus.DOWNLOAD_AUTHORIZED:
        raise PermissionError(
            f"{source.get('source_id', 'unknown')}: status must be DOWNLOAD_AUTHORIZED"
        )
    if not source.get("license") or source.get("license") in {"UNKNOWN", "NOT VERIFIED"}:
        raise PermissionError("verified license is required before download")
    if "download" not in source.get("allowed_operations", []):
        raise PermissionError("registry must explicitly allow download")


def require_training_authorized(source: dict) -> None:
    """Allow fitting only for reviewed data with an explicit training grant."""
    if source.get("status") != DatasetStatus.APPROVED_FOR_TRAINING:
        raise PermissionError(
            f"{source.get('source_id', 'unknown')}: status must be APPROVED_FOR_TRAINING"
        )
    if not source.get("license") or source.get("license") in {"UNKNOWN", "NOT VERIFIED"}:
        raise PermissionError("verified license is required before training")
    if "training" not in source.get("allowed_operations", []):
        raise PermissionError("registry must explicitly allow training")


def validate_bbox(box: list[float] | tuple[float, ...], *, normalized: bool = True) -> None:
    if len(box) != 4:
        raise ValueError("bbox must contain x1, y1, x2, y2")
    x1, y1, x2, y2 = map(float, box)
    if not x1 < x2 or not y1 < y2:
        raise ValueError("bbox must have positive area")
    if normalized and not all(0.0 <= value <= 1.0 for value in (x1, y1, x2, y2)):
        raise ValueError("normalized bbox coordinates must be in [0, 1]")


@dataclass(frozen=True)
class UnifiedAnnotation:
    source_id: str
    source_asset_id: str
    source_hash: str
    original_label: str
    mapping_version: str
    review_status: str
    reviewer_id: str | None = None
    subject_id: str | None = None
    session_id: str | None = None
    video_id: str | None = None
    camera_id: str | None = None
    start_timestamp: float | None = None
    end_timestamp: float | None = None
    frame_rate: float | None = None
    phone_visibility: str = "UNKNOWN_PHONE_VISIBILITY"
    phone_action: str = "UNKNOWN_ACTION"
    phone_context: str = "UNKNOWN_CONTEXT"
    occupant_role: str = "UNKNOWN"
    seatbelt: str = "UNCERTAIN_OR_OCCLUDED"
    bbox_xyxy_normalized: tuple[float, float, float, float] | None = None
    annotation_confidence: float | None = None
    license_reference: str | None = None

    def validate(self) -> None:
        if not self.source_id or not self.source_asset_id or len(self.source_hash) < 32:
            raise ValueError("source identity and hash are required")
        if self.phone_visibility not in PHONE_VISIBILITY:
            raise ValueError("invalid phone visibility")
        if self.phone_action not in PHONE_ACTION:
            raise ValueError("invalid phone action")
        if self.phone_context not in PHONE_CONTEXT:
            raise ValueError("invalid phone context")
        if self.occupant_role not in OCCUPANT_ROLES or self.seatbelt not in SEATBELT:
            raise ValueError("invalid occupant role or seatbelt state")
        if self.bbox_xyxy_normalized is not None:
            validate_bbox(self.bbox_xyxy_normalized)
        if self.phone_visibility == "PHONE_BOX" and self.bbox_xyxy_normalized is None:
            raise ValueError("PHONE_BOX requires bbox coordinates")
        timestamps = (self.start_timestamp, self.end_timestamp)
        if any(value is not None for value in timestamps):
            if not self.video_id or any(value is None for value in timestamps):
                raise ValueError("timestamps require a video and complete interval")
            if self.start_timestamp < 0 or self.end_timestamp <= self.start_timestamp:
                raise ValueError("invalid event interval")
        if self.review_status == "APPROVED" and not self.reviewer_id:
            raise ValueError("approved annotations require an authenticated reviewer")
        if self.phone_action == "SAFE_DRIVING" and self.phone_visibility in {
            "PHONE_VISIBLE",
            "PHONE_OCCLUDED",
            "PHONE_BOX",
        }:
            raise ValueError(
                "visible phone evidence cannot be automatically mapped to SAFE_DRIVING"
            )


def validate_group_split(records: list[dict]) -> dict:
    """Reject source, group, subject, video or augmented-origin leakage."""
    dimensions = (
        "sha256",
        "source_group_id",
        "effective_group_id",
        "trusted_subject_id",
        "video_id",
        "original_asset_id",
    )
    overlaps: dict[str, list[str]] = {}
    for dimension in dimensions:
        seen: defaultdict[str, set[str]] = defaultdict(set)
        for record in records:
            value, split = record.get(dimension), record.get("split")
            if value and split:
                seen[str(value)].add(str(split))
        overlaps[dimension] = sorted(value for value, splits in seen.items() if len(splits) > 1)
    protected = sorted(
        {
            str(record.get("trusted_subject_id"))
            for record in records
            if record.get("split") in {"train", "val"}
            and record.get("trusted_subject_id") in PROTECTED_SUBJECTS
        }
    )
    status = "PASS" if not any(overlaps.values()) and not protected else "FAIL"
    return {"status": status, "overlaps": overlaps, "protected_subjects_in_fit": protected}
