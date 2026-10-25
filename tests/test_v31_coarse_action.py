from __future__ import annotations

import numpy as np

from training.dmd.run_v31_coarse_action import (
    decode_coarse_phone_events,
    gate_object_events,
    recover_object_events,
    remap_coarse_labels,
)
from training.dmd.v30_causal_action import ActionDecoderConfig


def test_coarse_label_mapping_preserves_unknown() -> None:
    labels = np.asarray([-1, 0, 1, 2, 3, 4, 5], dtype=np.int64)
    assert remap_coarse_labels(labels).tolist() == [-1, 0, 1, 1, 1, 1, 2]


def test_coarse_decoder_is_causal_and_merges_short_gap() -> None:
    timestamps = np.arange(0.0, 3.0, 0.5)
    probabilities = np.asarray(
        [
            [0.9, 0.05, 0.05],
            [0.05, 0.9, 0.05],
            [0.05, 0.9, 0.05],
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
            [0.9, 0.05, 0.05],
        ],
        dtype=np.float32,
    )
    events = decode_coarse_phone_events(
        timestamps,
        probabilities,
        ActionDecoderConfig(threshold=0.45, release_patience=1),
    )
    assert len(events) == 1
    assert events[0]["start_seconds"] == 0.5
    assert events[0]["end_seconds"] == 1.5
    assert events[0]["source"] == "V31_COARSE_CAUSAL_ACTION"


def test_object_gating_and_preregistered_recovery_are_evidence_bounded() -> None:
    objects = [
        {"start_seconds": 1.0, "end_seconds": 3.0, "confidence": 0.5},
        {"start_seconds": 8.0, "end_seconds": 9.0, "confidence": 0.5},
    ]
    actions = [
        {"start_seconds": 2.0, "end_seconds": 4.0, "duration": 2.0, "confidence": 0.8},
        {"start_seconds": 5.0, "end_seconds": 6.5, "duration": 1.5, "confidence": 0.75},
        {"start_seconds": 10.0, "end_seconds": 12.0, "duration": 2.0, "confidence": 0.6},
    ]
    assert len(gate_object_events(objects, actions)) == 1
    recovered = recover_object_events(
        objects,
        actions,
        minimum_confidence=0.7,
        minimum_duration=1.0,
        merge_gap_seconds=0.6,
    )
    assert [(row["start_seconds"], row["end_seconds"]) for row in recovered] == [
        (1.0, 3.0),
        (5.0, 6.5),
        (8.0, 9.0),
    ]
