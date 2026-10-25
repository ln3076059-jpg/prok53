from __future__ import annotations

import numpy as np
import pytest

from backend.ai.events import Observation, TemporalEventEngine
from training.dmd.run_autonomous_v21 import RawVideoBoundaryCollector
from training.dmd.v31_streaming_fusion import (
    StreamingFusionConfig,
    StreamingObjectActionFusion,
)


def _row(class_id: int, confidence: float = 0.9) -> np.ndarray:
    row = np.full(6, (1.0 - confidence) / 5.0, dtype=np.float64)
    row[class_id] = confidence
    return row


def _run(rows: list[tuple[float, bool, int]]) -> list[dict]:
    fusion = StreamingObjectActionFusion(
        StreamingFusionConfig(
            action_threshold=0.55,
            label_update_margin=0.05,
            probability_ema_alpha=0.8,
        )
    )
    for timestamp, active, class_id in rows:
        fusion.update(
            timestamp=timestamp,
            object_active=active,
            object_confidence=0.8 if active else 0.0,
            action_probabilities=_row(class_id),
        )
    return fusion.decisions()


def test_streaming_fusion_decisions_are_prefix_invariant() -> None:
    prefix = [(0.0, False, 0), (0.5, True, 1), (1.0, True, 1)]
    future = [(1.5, True, 2), (2.0, False, 0)]
    prefix_decisions = _run(prefix)
    full_decisions = _run(prefix + future)
    assert full_decisions[: len(prefix_decisions)] == prefix_decisions
    assert [row["decision_type"] for row in full_decisions] == [
        "START",
        "LABEL_UPDATE",
        "END",
    ]
    assert full_decisions[1]["decision_time_seconds"] == 1.5


def test_streaming_fusion_records_decision_time_without_backdating() -> None:
    decisions = _run([(0.0, False, 0), (0.5, True, 1), (1.0, False, 0)])
    assert [row["decision_type"] for row in decisions] == ["START", "END"]
    assert decisions[0]["decision_time_seconds"] == 0.5
    assert decisions[0]["event_start_seconds"] == 0.5
    assert decisions[1]["decision_time_seconds"] == 1.0
    assert decisions[1]["event_end_seconds"] == 1.0


def test_streaming_fusion_separates_backdated_event_start_from_decision_time() -> None:
    fusion = StreamingObjectActionFusion(
        StreamingFusionConfig(action_threshold=0.55)
    )
    emitted = fusion.update(
        timestamp=2.0,
        object_active=True,
        object_confidence=0.8,
        action_probabilities=_row(1),
        object_event_start_seconds=1.5,
    )
    assert emitted[0].event_start_seconds == 1.5
    assert emitted[0].decision_time_seconds == 2.0


def test_streaming_fusion_rejects_object_start_from_the_future() -> None:
    fusion = StreamingObjectActionFusion()
    with pytest.raises(ValueError, match="cannot be later"):
        fusion.update(
            timestamp=2.0,
            object_active=True,
            object_confidence=0.8,
            action_probabilities=_row(1),
            object_event_start_seconds=2.1,
        )


def test_action_signal_cannot_open_event_without_object_state() -> None:
    decisions = _run([(0.0, False, 1), (0.5, False, 1), (1.0, False, 1)])
    assert decisions == []


def test_streaming_fusion_rejects_non_monotonic_timestamps() -> None:
    fusion = StreamingObjectActionFusion()
    fusion.update(
        timestamp=1.0,
        object_active=False,
        object_confidence=0.0,
        action_probabilities=_row(0),
    )
    with pytest.raises(ValueError, match="strictly increasing"):
        fusion.update(
            timestamp=1.0,
            object_active=False,
            object_confidence=0.0,
            action_probabilities=_row(0),
        )


def test_action_only_activation_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="not authorized"):
        StreamingFusionConfig(allow_action_only_activation=True)


def test_raw_boundary_collector_records_append_only_live_decisions() -> None:
    engine = TemporalEventEngine(
        window_seconds=2.0,
        min_positive_seconds=0.0,
        min_observations=2,
        cooldown_seconds=0.0,
        feature_positive_score=0.2,
        candidate_threshold=0.2,
        activation_threshold=0.2,
        release_threshold=0.15,
    )
    collector = RawVideoBoundaryCollector(engine)

    def observation(timestamp: float, score: float) -> Observation:
        return Observation(
            timestamp=timestamp,
            class_name="phone",
            confidence=score,
            fusion_score=score,
            track_id=1,
            occupant_role="driver",
            vehicle_context_id="veh1",
            phone_context="HANDHELD_USE" if score > 0 else "UNKNOWN",
            evidence_source=("AUTONOMOUS_V21" if score > 0 else "AUTONOMOUS_V21_REJECTED"),
        )

    collector.add(observation(1.0, 0.9))
    collector.add(observation(1.5, 0.9))
    prefix = [dict(row) for row in collector.decision_log]
    assert [row["decision_type"] for row in prefix] == ["START"]
    assert prefix[0]["decision_time_seconds"] == 1.5
    assert prefix[0]["event_start_seconds"] == 1.0

    collector.add(observation(2.0, 0.0))
    assert collector.decision_log[: len(prefix)] == prefix
    assert [row["decision_type"] for row in collector.decision_log] == [
        "START",
        "END",
    ]
    assert collector.decision_log[1]["decision_time_seconds"] == 2.0
    assert collector.decision_log[1]["event_end_seconds"] == 2.0
    assert collector.decision_log[1]["reason"] == "EVIDENCE_RELEASE_OR_EXPIRY"


def test_raw_boundary_collector_finalizes_open_event_at_session_boundary() -> None:
    engine = TemporalEventEngine(
        window_seconds=2.0,
        min_positive_seconds=0.0,
        min_observations=2,
        cooldown_seconds=0.0,
        feature_positive_score=0.2,
        candidate_threshold=0.2,
        activation_threshold=0.2,
        release_threshold=0.15,
    )
    collector = RawVideoBoundaryCollector(engine)
    for timestamp in (1.0, 1.5):
        collector.add(
            Observation(
                timestamp=timestamp,
                class_name="phone",
                confidence=0.9,
                fusion_score=0.9,
                track_id=1,
                occupant_role="driver",
                vehicle_context_id="veh1",
                phone_context="HANDHELD_USE",
            )
        )
    collector.finalize(1.5, reason="MAX_FRAMES")
    assert [row["decision_type"] for row in collector.decision_log] == [
        "START",
        "END",
    ]
    assert collector.decision_log[-1]["reason"] == "MAX_FRAMES"
    assert collector.decision_log[-1]["event_end_seconds"] == 1.5
