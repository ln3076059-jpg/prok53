"""Conservative driver-phone association with explicit UNKNOWN handling."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import StrEnum

from training.dmd.behavior_roi import Box


class AssociationState(StrEnum):
    DRIVER_ASSOCIATED = "DRIVER_ASSOCIATED"
    PASSENGER_ASSOCIATED = "PASSENGER_ASSOCIATED"
    MOUNTED_OR_STATIC = "MOUNTED_OR_STATIC"
    UNKNOWN = "UNKNOWN"


@dataclass(frozen=True)
class OccupantTrack:
    track_id: str
    role: str
    box: Box
    verified: bool
    confidence: float


@dataclass(frozen=True)
class AssociationResult:
    state: str
    confidence: float
    occupant_track_id: str
    source_evidence: tuple[str, ...]
    reason_codes: tuple[str, ...]

    def as_dict(self) -> dict:
        return asdict(self)


def intersection_over_box(target: Box, container: Box) -> float:
    x1 = max(target.x1, container.x1)
    y1 = max(target.y1, container.y1)
    x2 = min(target.x2, container.x2)
    y2 = min(target.y2, container.y2)
    area = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    return area / (target.width * target.height)


class DriverPhoneAssociation:
    def __init__(self, minimum_containment: float = 0.55):
        if not 0 < minimum_containment <= 1:
            raise ValueError("minimum containment must be in (0, 1]")
        self.minimum_containment = minimum_containment

    def associate(
        self,
        phone_box: Box,
        occupants: tuple[OccupantTrack, ...],
        *,
        phone_context: str,
        phone_confidence: float,
        hand_proximity: float | None = None,
        face_proximity: float | None = None,
    ) -> AssociationResult:
        if phone_context == "MOUNTED_OR_STATIC":
            return AssociationResult(
                state=AssociationState.MOUNTED_OR_STATIC.value,
                confidence=float(phone_confidence),
                occupant_track_id="UNKNOWN",
                source_evidence=("PHONE_CONTEXT",),
                reason_codes=("VERIFIED_MOUNTED_OR_STATIC_CONTEXT",),
            )
        verified = [track for track in occupants if track.verified]
        if not verified:
            return AssociationResult(
                state=AssociationState.UNKNOWN.value,
                confidence=0.0,
                occupant_track_id="UNKNOWN",
                source_evidence=("PHONE_BOX",),
                reason_codes=("NO_VERIFIED_OCCUPANT_TRACK",),
            )
        scored = [
            (intersection_over_box(phone_box, track.box), track) for track in verified
        ]
        overlap, track = max(scored, key=lambda item: item[0])
        if overlap < self.minimum_containment:
            return AssociationResult(
                state=AssociationState.UNKNOWN.value,
                confidence=float(overlap),
                occupant_track_id="UNKNOWN",
                source_evidence=("PHONE_BOX", "VERIFIED_OCCUPANT_TRACKS"),
                reason_codes=("NO_OCCUPANT_CONTAINS_PHONE",),
            )
        if track.role not in {"driver", "front_passenger", "rear_passenger"}:
            return AssociationResult(
                state=AssociationState.UNKNOWN.value,
                confidence=float(overlap * track.confidence),
                occupant_track_id=track.track_id,
                source_evidence=("PHONE_BOX", "OCCUPANT_TRACK"),
                reason_codes=("OCCUPANT_ROLE_UNKNOWN",),
            )
        geometry = max(hand_proximity or 0.0, face_proximity or 0.0)
        confidence = min(
            1.0,
            overlap
            * track.confidence
            * phone_confidence
            * (0.75 + 0.25 * geometry),
        )
        state = (
            AssociationState.DRIVER_ASSOCIATED
            if track.role == "driver"
            else AssociationState.PASSENGER_ASSOCIATED
        )
        return AssociationResult(
            state=state.value,
            confidence=float(confidence),
            occupant_track_id=track.track_id,
            source_evidence=("PHONE_BOX", "VERIFIED_OCCUPANT_TRACK", "ROLE"),
            reason_codes=("PHONE_CONTAINED_BY_VERIFIED_OCCUPANT",),
        )
