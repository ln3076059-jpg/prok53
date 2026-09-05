from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Observation:
    timestamp: float
    class_name: str
    confidence: float
    track_id: int | None
    occupant_role: str | None
    vehicle_context_id: str | None
    phone_context: str = "UNKNOWN"
    fusion_score: float | None = None
    evidence_source: str = "DETECTOR_ONLY"
    vehicle_type: str = "unknown"
    occupant_role_confidence: float = 0.0
    vehicle_context_confidence: float = 0.0
    phone_hand_proximity: float = 0.0
    phone_face_proximity: float = 0.0
    pose_confidence: float = 0.0
    seatbelt_probabilities: tuple[float, float, float] | None = None
    cabin_id: str | None = None
    visibility: str = ""
    outside_vehicle_person: str = ""
    behavior_label: str = ""
    occupant_id: str = ""


@dataclass(frozen=True)
class EventCandidate:
    event_type: str
    confidence: float
    timestamp: float
    start_timestamp: float
    end_timestamp: float
    observation_count: int
    track_id: int | None
    review_status: str
    occupant_role: str
    vehicle_context_id: str
    cabin_id: str = ""
    visibility: str = ""
    outside_vehicle_person: str = ""
    behavior_label: str = ""
    vehicle_type: str = "unknown"
    temporal_score: float = 0.0
    occupant_id: str = ""


class TemporalEventEngine:
    def __init__(
        self,
        window_seconds: float = 2.0,
        min_positive_seconds: float = 1.2,
        min_observations: int = 3,
        cooldown_seconds: float = 5.0,
        smoothing_alpha: float = 0.35,
        feature_positive_score: float = 0.5,
        positive_ratio: float = 0.60,
        gap_tolerance_seconds: float | None = None,
        candidate_threshold: float = 0.45,
        activation_threshold: float = 0.55,
        release_threshold: float = 0.35,
        **_: object,
    ):
        self.window_seconds = float(window_seconds)
        self.min_positive_seconds = float(min_positive_seconds)
        self.min_observations = int(min_observations)
        self.cooldown_seconds = float(cooldown_seconds)
        self.smoothing_alpha = float(smoothing_alpha)
        self.feature_positive_score = float(feature_positive_score)
        self.minimum_positive_ratio = float(positive_ratio)
        self.gap_tolerance_seconds = float(gap_tolerance_seconds or window_seconds)
        self.candidate_threshold = float(candidate_threshold)
        self.activation_threshold = float(activation_threshold)
        self.release_threshold = float(release_threshold)
        self.windows: defaultdict[tuple[str, int | None, str], deque[Observation]] = defaultdict(
            deque
        )
        self.last_event: dict[tuple[str, str, int | None, str], float] = {}
        self.smoothed: dict[tuple[str, int | None, str], float] = {}
        self.active_events: set[tuple[str, str, int | None, str]] = set()

    def add(self, observation: Observation) -> list[EventCandidate]:
        if observation.vehicle_context_id is None:
            return []
        role = observation.occupant_role or "unknown"
        if observation.class_name == "phone" and role != "driver":
            return []
        if observation.class_name == "phone" and observation.phone_context == "MOUNTED_OR_STATIC":
            self._release("PHONE", observation)
            self.windows.pop((observation.vehicle_context_id, observation.track_id, role), None)
            return []
        if (
            observation.class_name.startswith("seatbelt_")
            and observation.vehicle_type == "motorcycle"
        ):
            return []
        if observation.class_name.startswith("seatbelt_") and role == "unknown":
            return []
        if observation.class_name in {"seatbelt_uncertain", "uncertain_or_occluded"}:
            return []
        score = (
            observation.fusion_score
            if observation.fusion_score is not None
            else observation.confidence
        )
        if score <= self.release_threshold:
            event_type = "PHONE" if observation.class_name == "phone" else "NO_SEATBELT"
            self._release(event_type, observation)
        # Belt classes describe the same occupant ROI but a class flip can receive a new
        # tracker ID. Group belt evidence by vehicle+seat so contradictions remain visible.
        window_track_id = observation.track_id if observation.class_name == "phone" else None
        window = self.windows[(observation.vehicle_context_id, window_track_id, role)]
        window.append(observation)
        while window and observation.timestamp - window[0].timestamp > self.window_seconds:
            window.popleft()
        return self._evaluate(window, observation)

    def _evaluate(self, window: deque[Observation], current: Observation) -> list[EventCandidate]:
        fastened = [item for item in window if item.class_name == "seatbelt_fastened"]
        unfastened = [item for item in window if item.class_name == "seatbelt_unfastened"]
        phone = [item for item in window if item.class_name == "phone"]
        candidates: list[EventCandidate] = []
        if self._persistent(phone):
            role = current.occupant_role or "unknown"
            phone_status = (
                "PENDING"
                if role == "driver"
                and all(item.phone_context in {"HANDHELD", "HANDHELD_USE"} for item in phone)
                and all(item.evidence_source != "FUSION_REJECTED" for item in phone)
                else "NEEDS_REVIEW"
            )
            candidates.append(self._candidate("PHONE", phone, current, phone_status))
        if self._persistent(unfastened):
            conflicting = self._persistent(fastened)
            rejected = any(
                item.evidence_source in {"FUSION_REJECTED", "DETECTOR_CLASSIFIER_CONFLICT"}
                for item in unfastened
            )
            candidates.append(
                self._candidate(
                    "NO_SEATBELT",
                    unfastened,
                    current,
                    "NEEDS_REVIEW" if conflicting or rejected else "PENDING",
                )
            )
        if self._persistent(fastened):
            self._release("NO_SEATBELT", current)
        return [item for item in candidates if self._activation_allows(item)]

    def _persistent(self, observations: list[Observation]) -> bool:
        if len(observations) < self.min_observations:
            return False
        if observations[-1].timestamp - observations[0].timestamp < self.min_positive_seconds:
            return False
        if any(
            right.timestamp - left.timestamp > self.gap_tolerance_seconds
            for left, right in zip(observations, observations[1:], strict=False)
        ):
            return False
        scores = [
            item.fusion_score if item.fusion_score is not None else item.confidence
            for item in observations
        ]
        positive_ratio = sum(score >= self.feature_positive_score for score in scores) / len(scores)
        if positive_ratio < self.minimum_positive_ratio:
            return False
        key = (
            observations[-1].vehicle_context_id or "",
            observations[-1].track_id,
            observations[-1].occupant_role or "unknown",
        )
        previous = self.smoothed.get(key, scores[0])
        for score in scores[1:]:
            previous = self.smoothing_alpha * score + (1.0 - self.smoothing_alpha) * previous
        self.smoothed[key] = previous
        return (
            previous >= self.activation_threshold
            and sum(scores) / len(scores) >= self.candidate_threshold
        )

    def _candidate(
        self, event_type: str, observations: list[Observation], current: Observation, status: str
    ) -> EventCandidate:
        if current.vehicle_context_id is None:
            raise ValueError("event candidate requires vehicle context")
        role = current.occupant_role or "unknown"
        scores = [
            item.fusion_score if item.fusion_score is not None else item.confidence
            for item in observations
        ]

        for item in observations:
            if (
                item.cabin_id != observations[0].cabin_id
                or item.occupant_id != observations[0].occupant_id
                or item.visibility != observations[0].visibility
                or item.outside_vehicle_person != observations[0].outside_vehicle_person
                or item.behavior_label != observations[0].behavior_label
                or item.vehicle_type != observations[0].vehicle_type
            ):
                status = "NEEDS_REVIEW"
                break
        occupant_id = current.occupant_id or (f"{current.vehicle_context_id}:occupant:{role}")
        return EventCandidate(
            event_type=event_type,
            confidence=sum(scores) / len(scores),
            timestamp=current.timestamp,
            start_timestamp=observations[0].timestamp,
            end_timestamp=observations[-1].timestamp,
            observation_count=len(observations),
            track_id=current.track_id,
            review_status=status,
            occupant_role=role,
            vehicle_context_id=current.vehicle_context_id,
            cabin_id=current.cabin_id or "",
            visibility=current.visibility,
            outside_vehicle_person=current.outside_vehicle_person,
            behavior_label=current.behavior_label,
            vehicle_type=current.vehicle_type,
            temporal_score=self.smoothed.get(
                (current.vehicle_context_id, current.track_id, role), scores[-1]
            ),
            occupant_id=occupant_id,
        )

    def _cooldown_allows(self, candidate: EventCandidate) -> bool:
        key = self._candidate_key(candidate)
        if candidate.timestamp - self.last_event.get(key, float("-inf")) < self.cooldown_seconds:
            return False
        self.last_event[key] = candidate.timestamp
        return True

    def _activation_allows(self, candidate: EventCandidate) -> bool:
        key = self._candidate_key(candidate)
        if key in self.active_events or not self._cooldown_allows(candidate):
            return False
        self.active_events.add(key)
        return True

    @staticmethod
    def _candidate_key(candidate: EventCandidate) -> tuple[str, str, int | None, str]:
        return (
            candidate.event_type,
            candidate.vehicle_context_id,
            candidate.track_id if candidate.event_type == "PHONE" else None,
            candidate.occupant_role,
        )

    def _release(self, event_type: str, observation: Observation) -> None:
        if observation.vehicle_context_id is None:
            return
        key = (
            event_type,
            observation.vehicle_context_id,
            observation.track_id if event_type == "PHONE" else None,
            observation.occupant_role or "unknown",
        )
        self.active_events.discard(key)

    def reset_vehicle(self, vehicle_context_id: str) -> None:
        for key in [key for key in self.windows if key[0] == vehicle_context_id]:
            del self.windows[key]
            self.smoothed.pop(key, None)
        for key in [key for key in self.last_event if key[1] == vehicle_context_id]:
            del self.last_event[key]
        self.active_events = {key for key in self.active_events if key[1] != vehicle_context_id}

    def expire(self, timestamp: float) -> None:
        """Remove stale state when a vehicle or behavior track disappears."""
        stale: list[tuple[str, int | None, str]] = []
        for key, window in self.windows.items():
            while window and timestamp - window[0].timestamp > self.window_seconds:
                window.popleft()
            if not window:
                stale.append(key)
        for key in stale:
            del self.windows[key]
            self.smoothed.pop(key, None)
            vehicle_context_id, track_id, role = key
            self.active_events.discard(("PHONE", vehicle_context_id, track_id, role))
            self.active_events.discard(("NO_SEATBELT", vehicle_context_id, None, role))
