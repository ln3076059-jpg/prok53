import pytest

from backend.ai.phone_tracker import (
    PhoneEvidence,
    PhoneEvidenceState,
    PhoneTrackletManager,
)


def _evidence(
    timestamp: float,
    box=(10.0, 10.0, 30.0, 30.0),
    *,
    associated: bool,
    handheld: bool,
    source: str = "auxiliary_roi",
) -> PhoneEvidence:
    return PhoneEvidence(
        timestamp=timestamp,
        box=box,
        confidence=0.5,
        source=source,
        driver_associated=associated,
        handheld_verified=handheld,
    )


def test_unverified_track_cannot_open_or_continue_event() -> None:
    manager = PhoneTrackletManager()

    tracked = manager.update(
        _evidence(1.0, associated=False, handheld=False),
        frame_width=100,
        frame_height=100,
    )

    assert tracked.state is PhoneEvidenceState.UNVERIFIED
    assert tracked.may_open_event is False
    assert tracked.may_continue_event is False
    assert tracked.should_refine_sampling is True


def test_verified_detection_promotes_track_and_may_open_event() -> None:
    manager = PhoneTrackletManager()

    tracked = manager.update(
        _evidence(1.0, associated=True, handheld=True),
        frame_width=100,
        frame_height=100,
    )

    assert tracked.state is PhoneEvidenceState.VERIFIED
    assert tracked.may_open_event is True
    assert tracked.may_continue_event is True


def test_nearby_uncertain_detection_only_continues_recent_verified_track() -> None:
    manager = PhoneTrackletManager(continuation_ttl_seconds=1.0)
    verified = manager.update(
        _evidence(1.0, associated=True, handheld=True),
        frame_width=100,
        frame_height=100,
    )
    continuation = manager.update(
        _evidence(1.5, box=(12.0, 11.0, 32.0, 31.0), associated=False, handheld=False),
        frame_width=100,
        frame_height=100,
    )

    assert continuation.track_id == verified.track_id
    assert continuation.state is PhoneEvidenceState.CONTINUATION
    assert continuation.may_open_event is False
    assert continuation.may_continue_event is True


def test_distant_uncertain_detection_does_not_inherit_verified_state() -> None:
    manager = PhoneTrackletManager(maximum_center_distance=0.1)
    verified = manager.update(
        _evidence(1.0, associated=True, handheld=True),
        frame_width=100,
        frame_height=100,
    )
    distant = manager.update(
        _evidence(
            1.2,
            box=(75.0, 75.0, 95.0, 95.0),
            associated=False,
            handheld=False,
        ),
        frame_width=100,
        frame_height=100,
    )

    assert distant.track_id != verified.track_id
    assert distant.state is PhoneEvidenceState.UNVERIFIED


def test_continuation_expires_by_timestamp_not_frame_count() -> None:
    manager = PhoneTrackletManager(
        track_ttl_seconds=2.0,
        continuation_ttl_seconds=0.5,
    )
    manager.update(
        _evidence(1.0, associated=True, handheld=True),
        frame_width=100,
        frame_height=100,
    )
    late = manager.update(
        _evidence(1.75, associated=False, handheld=False),
        frame_width=100,
        frame_height=100,
    )

    assert late.state is PhoneEvidenceState.UNVERIFIED
    manager.advance(4.0)
    assert manager.active_track_count == 0


def test_invalid_ttl_configuration_is_rejected() -> None:
    with pytest.raises(ValueError, match="no greater than track TTL"):
        PhoneTrackletManager(track_ttl_seconds=1.0, continuation_ttl_seconds=2.0)
