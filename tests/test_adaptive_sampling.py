import pytest

from training.dmd.adaptive_sampling import AdaptiveFrameSampler


def test_adaptive_sampler_uses_dense_stride_during_trigger_window() -> None:
    sampler = AdaptiveFrameSampler(base_stride=15, dense_stride=3, hold_seconds=2.0)

    assert sampler.next_stride(0.0, trigger=False) == 15
    assert sampler.next_stride(1.0, trigger=True) == 3
    assert sampler.next_stride(2.5, trigger=False) == 3
    assert sampler.next_stride(3.01, trigger=False) == 15


def test_adaptive_trigger_extends_existing_dense_window() -> None:
    sampler = AdaptiveFrameSampler(base_stride=15, dense_stride=3, hold_seconds=2.0)
    sampler.observe(1.0, trigger=True)
    sampler.observe(2.5, trigger=True)

    assert sampler.is_dense(4.4) is True
    assert sampler.is_dense(4.6) is False


def test_adaptive_sampler_reset_returns_to_base_stride() -> None:
    sampler = AdaptiveFrameSampler(base_stride=15, dense_stride=3, hold_seconds=2.0)
    sampler.observe(1.0, trigger=True)
    sampler.reset()

    assert sampler.next_stride(1.1, trigger=False) == 15


def test_absolute_scheduler_preserves_every_baseline_frame_after_dense_burst() -> None:
    sampler = AdaptiveFrameSampler(base_stride=15, dense_stride=3, hold_seconds=0.4)
    fps = 30.0
    sampled = [0]
    current = 0
    while current < 90:
        current = sampler.next_frame_index(
            current,
            current / fps,
            fps,
            trigger=current == 15,
        )
        sampled.append(current)

    assert set(range(0, 91, 15)).issubset(sampled)
    assert {18, 21, 24, 27}.issubset(sampled)
    assert sampled == sorted(set(sampled))


def test_absolute_scheduler_rejects_invalid_frame_inputs() -> None:
    sampler = AdaptiveFrameSampler(base_stride=15, dense_stride=3, hold_seconds=2.0)
    with pytest.raises(ValueError, match="frame_index"):
        sampler.next_frame_index(-1, 0.0, 30.0, trigger=False)
    with pytest.raises(ValueError, match="fps"):
        sampler.next_frame_index(0, 0.0, 0.0, trigger=False)


def test_adaptive_sampler_rejects_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="must not exceed"):
        AdaptiveFrameSampler(base_stride=3, dense_stride=15, hold_seconds=2.0)
