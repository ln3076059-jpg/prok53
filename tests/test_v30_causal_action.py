from __future__ import annotations

import numpy as np
import pytest
import torch

from training.dmd.v30_causal_action import (
    ActionDecoderConfig,
    CausalActionGRU,
    causal_clip_indices,
    class_aware_event_metrics,
    decode_action_events,
)


def test_causal_windows_end_at_their_target_without_future_indices() -> None:
    windows = causal_clip_indices(12, sequence_length=5, stride=2)
    assert [row.tolist() for row in windows] == [
        [0, 1, 2, 3, 4],
        [2, 3, 4, 5, 6],
        [4, 5, 6, 7, 8],
        [6, 7, 8, 9, 10],
    ]
    assert all(int(row.max()) == int(row[-1]) for row in windows)


def test_causal_action_gru_is_unidirectional_and_has_six_outputs() -> None:
    model = CausalActionGRU(embedding_dim=8, hidden_size=4)
    assert model.gru.bidirectional is False
    assert model(torch.randn(2, 5, 8)).shape == (2, 6)


def test_causal_action_gru_supports_preregistered_coarse_head() -> None:
    model = CausalActionGRU(embedding_dim=8, hidden_size=4, num_classes=3)
    assert model.gru.bidirectional is False
    assert model(torch.randn(2, 5, 8)).shape == (2, 3)


def test_decoder_splits_texting_to_phonecall_transition() -> None:
    timestamps = np.arange(0.0, 3.0, 0.5)
    probabilities = np.full((len(timestamps), 6), 0.01, dtype=np.float32)
    probabilities[:3, 1] = 0.90
    probabilities[3:, 3] = 0.90
    events = decode_action_events(
        timestamps,
        probabilities,
        ActionDecoderConfig(threshold=0.45, release_patience=1),
    )
    assert [row["source_action"] for row in events] == [
        "driver_actions/texting_left",
        "driver_actions/phonecall_left",
    ]
    assert events[0]["end_seconds"] == pytest.approx(events[1]["start_seconds"])


def test_class_aware_matcher_rejects_temporally_correct_wrong_action() -> None:
    truth = [
        {
            "source_action": "driver_actions/texting_left",
            "start_seconds": 1.0,
            "end_seconds": 4.0,
        }
    ]
    prediction = [
        {
            "source_action": "driver_actions/phonecall_left",
            "start_seconds": 1.0,
            "end_seconds": 4.0,
        }
    ]
    metrics = class_aware_event_metrics(truth, prediction, duration_seconds=10.0)
    assert metrics["true_positives"] == 0
    assert metrics["false_positives"] == 1
    assert metrics["false_negatives"] == 1
