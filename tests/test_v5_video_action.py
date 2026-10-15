import numpy as np
import torch

from training.dmd.v5_resnet_gru import ResNetGRUClassifier
from training.dmd.v5_video_action import (
    ACTION_CLASSES,
    merge_evidence_events,
    probabilities_to_events,
)


def test_resnet_gru_temporal_head_shape():
    model = ResNetGRUClassifier()
    logits = model(torch.zeros(2, 9, 512))

    assert logits.shape == (2, len(ACTION_CLASSES))


def test_probabilities_to_events_requires_temporal_persistence():
    timestamps = np.arange(7, dtype=float) * 0.5
    probabilities = np.zeros((7, len(ACTION_CLASSES)), dtype=float)
    probabilities[:, 0] = 1.0
    probabilities[2, 0] = 0.1
    probabilities[2, 1] = 0.9

    assert probabilities_to_events(timestamps, probabilities, threshold=0.6) == []


def test_action_only_event_is_needs_review():
    timestamps = np.arange(8, dtype=float) * 0.5
    probabilities = np.zeros((8, len(ACTION_CLASSES)), dtype=float)
    probabilities[:, 0] = 1.0
    probabilities[2:6, 0] = 0.05
    probabilities[2:6, 1] = 0.95

    events = probabilities_to_events(timestamps, probabilities, threshold=0.6)

    assert len(events) == 1
    assert events[0]["source"] == "VIDEO_ACTION"
    assert events[0]["review_status"] == "NEEDS_REVIEW"


def test_object_and_action_fusion_preserves_governance_status():
    object_events = [
        {"start_seconds": 10.0, "end_seconds": 12.0, "confidence": 0.4, "event_type": "PHONE"}
    ]
    action_events = [
        {
            "start_seconds": 11.0,
            "end_seconds": 14.0,
            "confidence": 0.8,
            "event_type": "PHONE",
            "source": "VIDEO_ACTION",
            "review_status": "NEEDS_REVIEW",
        }
    ]

    fused = merge_evidence_events(object_events, action_events)

    assert len(fused) == 1
    assert fused[0]["sources"] == ["OBJECT_GEOMETRY", "VIDEO_ACTION"]
    assert fused[0]["review_status"] == "PENDING"
