"""Guard against attributing every sub-second miss to the temporal minimum."""

from training.dmd.audit_v26_optimization import classify_fn


def test_two_samples_over_half_second_without_proposals_is_not_temporal_failure():
    cause, _ = classify_fn(
        {
            "matched": False,
            "duration_seconds": 0.9072,
            "sampled_frames": 2,
            "sampled_span_seconds": 0.504,
            "candidate_frames": 0,
            "driver_verified_frames": 2,
        }
    )
    assert cause == "NO_PHONE_PROPOSAL_IN_CACHED_TRACE"


def test_single_sample_cannot_supply_two_observation_support():
    cause, _ = classify_fn(
        {
            "matched": False,
            "duration_seconds": 0.8064,
            "sampled_frames": 1,
            "sampled_span_seconds": 0.0,
        }
    )
    assert cause == "TEMPORAL_SUPPORT_BELOW_ENGINE_MINIMUM"
