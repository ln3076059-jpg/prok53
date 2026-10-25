"""Fail-closed phone tracklets for preserving identity without fabricating violations."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from math import hypot


BoxTuple = tuple[float, float, float, float]


class PhoneEvidenceState(str, Enum):
    UNVERIFIED = "UNVERIFIED"
    VERIFIED = "VERIFIED"
    CONTINUATION = "CONTINUATION"


def box_iou(left: BoxTuple, right: BoxTuple) -> float:
    x1 = max(left[0], right[0])
    y1 = max(left[1], right[1])
    x2 = min(left[2], right[2])
    y2 = min(left[3], right[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    left_area = max(0.0, left[2] - left[0]) * max(0.0, left[3] - left[1])
    right_area = max(0.0, right[2] - right[0]) * max(0.0, right[3] - right[1])
    union = left_area + right_area - intersection
    return intersection / union if union > 0.0 else 0.0


@dataclass(frozen=True)
class PhoneEvidence:
    timestamp: float
    box: BoxTuple
    confidence: float
    source: str
    driver_associated: bool
    handheld_verified: bool


@dataclass
class _PhoneTrack:
    track_id: int
    box: BoxTuple
    last_timestamp: float
    last_verified_timestamp: float | None
    sources: set[str] = field(default_factory=set)


@dataclass(frozen=True)
class TrackedPhoneEvidence:
    track_id: int
    state: PhoneEvidenceState
    evidence: PhoneEvidence
    seconds_since_verified: float | None
    source_evidence: tuple[str, ...]

    @property
    def may_open_event(self) -> bool:
        return self.state is PhoneEvidenceState.VERIFIED

    @property
    def may_continue_event(self) -> bool:
        return self.state in {
            PhoneEvidenceState.VERIFIED,
            PhoneEvidenceState.CONTINUATION,
        }

    @property
    def should_refine_sampling(self) -> bool:
        return True


class PhoneTrackletManager:
    """Associate phone evidence while keeping uncertain evidence fail-closed.

    A verified detection must be both driver-associated and handheld-verified. An
    uncertain detection can continue a recent verified track, but it can never open
    a violation by itself. Track expiry is timestamp-based so behavior is stable
    across different source frame rates and sampling strides.
    """

    def __init__(
        self,
        *,
        track_ttl_seconds: float = 2.0,
        continuation_ttl_seconds: float = 1.0,
        minimum_iou: float = 0.05,
        maximum_center_distance: float = 0.12,
    ) -> None:
        if track_ttl_seconds <= 0.0:
            raise ValueError("track_ttl_seconds must be positive")
        if not 0.0 < continuation_ttl_seconds <= track_ttl_seconds:
            raise ValueError(
                "continuation_ttl_seconds must be positive and no greater than track TTL"
            )
        self.track_ttl_seconds = float(track_ttl_seconds)
        self.continuation_ttl_seconds = float(continuation_ttl_seconds)
        self.minimum_iou = float(minimum_iou)
        self.maximum_center_distance = float(maximum_center_distance)
        self._tracks: dict[int, _PhoneTrack] = {}
        self._next_track_id = 1

    def _expire(self, timestamp: float) -> None:
        expired = [
            track_id
            for track_id, track in self._tracks.items()
            if timestamp - track.last_timestamp > self.track_ttl_seconds
        ]
        for track_id in expired:
            del self._tracks[track_id]

    @staticmethod
    def _center(box: BoxTuple) -> tuple[float, float]:
        return ((box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0)

    def _match(
        self,
        evidence: PhoneEvidence,
        frame_width: int,
        frame_height: int,
    ) -> _PhoneTrack | None:
        diagonal = max(hypot(frame_width, frame_height), 1.0)
        center_x, center_y = self._center(evidence.box)
        candidates: list[tuple[float, _PhoneTrack]] = []
        for track in self._tracks.values():
            track_x, track_y = self._center(track.box)
            distance = hypot(center_x - track_x, center_y - track_y) / diagonal
            overlap = box_iou(evidence.box, track.box)
            if overlap < self.minimum_iou and distance > self.maximum_center_distance:
                continue
            recency = 1.0 - min(
                1.0,
                (evidence.timestamp - track.last_timestamp) / self.track_ttl_seconds,
            )
            candidates.append(
                (0.65 * overlap + 0.25 * (1.0 - distance) + 0.10 * recency, track)
            )
        return max(candidates, key=lambda item: item[0])[1] if candidates else None

    def update(
        self,
        evidence: PhoneEvidence,
        *,
        frame_width: int,
        frame_height: int,
    ) -> TrackedPhoneEvidence:
        self._expire(evidence.timestamp)
        track = self._match(evidence, frame_width, frame_height)
        if track is None:
            track = _PhoneTrack(
                track_id=self._next_track_id,
                box=evidence.box,
                last_timestamp=evidence.timestamp,
                last_verified_timestamp=None,
            )
            self._tracks[track.track_id] = track
            self._next_track_id += 1

        is_verified = evidence.driver_associated and evidence.handheld_verified
        seconds_since_verified = (
            evidence.timestamp - track.last_verified_timestamp
            if track.last_verified_timestamp is not None
            else None
        )
        if is_verified:
            state = PhoneEvidenceState.VERIFIED
            track.last_verified_timestamp = evidence.timestamp
            seconds_since_verified = 0.0
        elif (
            seconds_since_verified is not None
            and 0.0 <= seconds_since_verified <= self.continuation_ttl_seconds
        ):
            state = PhoneEvidenceState.CONTINUATION
        else:
            state = PhoneEvidenceState.UNVERIFIED

        track.box = evidence.box
        track.last_timestamp = evidence.timestamp
        track.sources.add(evidence.source)
        return TrackedPhoneEvidence(
            track_id=track.track_id,
            state=state,
            evidence=evidence,
            seconds_since_verified=seconds_since_verified,
            source_evidence=tuple(sorted(track.sources)),
        )

    def advance(self, timestamp: float) -> None:
        self._expire(timestamp)

    def reset(self) -> None:
        self._tracks.clear()

    @property
    def active_track_count(self) -> int:
        return len(self._tracks)
