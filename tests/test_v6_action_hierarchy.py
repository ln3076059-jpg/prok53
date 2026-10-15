import numpy as np
import pytest
import torch

from training.dmd.v5_resnet_gru import EmbeddingClipDataset, SubjectEmbeddings, train_resnet_gru
from training.dmd.v6_action_hierarchy import (
    HierarchicalResNetGRUClassifier,
    coarse_targets,
    hierarchical_probabilities,
)


def _dataset() -> EmbeddingClipDataset:
    subject = SubjectEmbeddings(
        subject_id="synthetic-development-subject",
        embeddings=np.arange(20 * 512, dtype=np.float32).reshape(20, 512) / 1000.0,
        timestamps=np.arange(20, dtype=np.float64) * 0.5,
        labels=np.asarray([0, 1, 2, 3, 4, 5, 0, 1, 2, 3] * 2, dtype=np.int64),
    )
    return EmbeddingClipDataset(
        [subject], sequence_length=3, stride=1, labeled_only=True
    )


def test_coarse_targets_preserve_safe_phone_other_semantics():
    labels = torch.tensor([0, 1, 2, 3, 4, 5])
    assert coarse_targets(labels).tolist() == [0, 1, 1, 1, 1, 2]


def test_hierarchical_probabilities_sum_to_one_in_flat_class_order():
    coarse_logits = torch.tensor([[2.0, 1.0, 0.0], [0.0, 2.0, 1.0]])
    phone_logits = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 1.0]])
    probabilities = hierarchical_probabilities(coarse_logits, phone_logits)

    assert probabilities.shape == (2, 6)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
    assert probabilities[0, 0] > probabilities[0, 1:5].sum()
    assert probabilities[1, 1:5].sum() > probabilities[1, 0]


def test_hierarchical_model_shapes_both_heads():
    model = HierarchicalResNetGRUClassifier(hidden_size=8)
    coarse, phone = model(torch.randn(4, 9, 512))
    assert coarse.shape == (4, 3)
    assert phone.shape == (4, 4)


def test_flat_training_rejects_unknown_balancing_mode():
    with pytest.raises(ValueError, match="unsupported sampling mode"):
        train_resnet_gru(
            _dataset(),
            epochs=1,
            batch_size=4,
            learning_rate=0.001,
            seed=1,
            sampling_mode="invalid",
        )
