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


@dataclass(frozen=True)
class EventCandidate:
    event_type: str
    confidence: float
    timestamp: float
    track_id: int | None
    review_status: str
    occupant_role: str
    vehicle_context_id: str


class TemporalEventEngine:
    def __init__(self, window_seconds: float = 2.0, min_positive_seconds: float = 1.2, min_observations: int = 3, cooldown_seconds: float = 5.0):
        self.window_seconds = window_seconds
        self.min_positive_seconds = min_positive_seconds
        self.min_observations = min_observations
        self.cooldown_seconds = cooldown_seconds
        self.windows: defaultdict[tuple[str, int | None, str], deque[Observation]] = defaultdict(deque)
        self.last_event: dict[tuple[str, str, int | None, str], float] = {}

    def add(self, observation: Observation) -> list[EventCandidate]:
        if observation.vehicle_context_id is None:
            return []
        if observation.occupant_role is None:
            return []
        if observation.class_name == "phone" and observation.occupant_role != "driver":
            return []
        if observation.class_name == "phone" and observation.phone_context == "MOUNTED_OR_STATIC":
            return []
        # Belt classes describe the same occupant ROI but a class flip can receive a new
        # tracker ID. Group belt evidence by vehicle+seat so contradictions remain visible.
        window_track_id = observation.track_id if observation.class_name == "phone" else None
        window = self.windows[
            (observation.vehicle_context_id, window_track_id, observation.occupant_role)
        ]
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
            phone_status = "PENDING" if all(item.phone_context == "HANDHELD" for item in phone) else "NEEDS_REVIEW"
            candidates.append(self._candidate("PHONE", phone, current, phone_status))
        if self._persistent(unfastened):
            conflicting = self._persistent(fastened)
            candidates.append(self._candidate("NO_SEATBELT", unfastened, current, "NEEDS_REVIEW" if conflicting else "PENDING"))
        return [item for item in candidates if self._cooldown_allows(item)]

    def _persistent(self, observations: list[Observation]) -> bool:
        if len(observations) < self.min_observations:
            return False
        return observations[-1].timestamp - observations[0].timestamp >= self.min_positive_seconds

    def _candidate(self, event_type: str, observations: list[Observation], current: Observation, status: str) -> EventCandidate:
        if current.occupant_role is None or current.vehicle_context_id is None:
            raise ValueError("event candidate requires vehicle context and occupant role")
        return EventCandidate(
            event_type,
            sum(item.confidence for item in observations) / len(observations),
            current.timestamp,
            current.track_id,
            status,
            current.occupant_role,
            current.vehicle_context_id,
        )

    def _cooldown_allows(self, candidate: EventCandidate) -> bool:
        key = (
            candidate.event_type,
            candidate.vehicle_context_id,
            candidate.track_id,
            candidate.occupant_role,
        )
        if candidate.timestamp - self.last_event.get(key, float("-inf")) < self.cooldown_seconds:
            return False
        self.last_event[key] = candidate.timestamp
        return True
