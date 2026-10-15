import numpy as np
import torch

from training.dmd.v5_resnet_gru import EmbeddingClipDataset, SubjectEmbeddings
from training.dmd.v7_action_models import (
    BinaryGateResNetGRUClassifier,
    binary_gate_probabilities,
    train_hierarchical_v7,
)


def _dataset() -> EmbeddingClipDataset:
    rng = np.random.default_rng(7)
    subject = SubjectEmbeddings(
        subject_id="synthetic-v7-subject",
        embeddings=rng.normal(size=(24, 512)).astype(np.float32),
        timestamps=np.arange(24, dtype=np.float64) * 0.5,
        labels=np.asarray([0, 1, 2, 3, 4, 5] * 4, dtype=np.int64),
    )
    return EmbeddingClipDataset(
        [subject], sequence_length=3, stride=1, labeled_only=True
    )


def test_binary_gate_probabilities_are_normalized_in_flat_order():
    binary = torch.tensor([[3.0, 0.0], [0.0, 3.0]])
    nonphone = torch.tensor([[2.0, 0.0], [0.0, 2.0]])
    phone = torch.tensor([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 2.0]])

    probabilities = binary_gate_probabilities(binary, nonphone, phone)

    assert probabilities.shape == (2, 6)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones(2), atol=1e-6)
    assert probabilities[0, 0] > probabilities[0, 5]
    assert probabilities[1, 4] == probabilities[1, 1:5].max()


def test_binary_gate_model_shapes_all_heads():
    model = BinaryGateResNetGRUClassifier(hidden_size=8)
    binary, nonphone, phone = model(torch.randn(4, 9, 512))
    assert binary.shape == (4, 2)
    assert nonphone.shape == (4, 2)
    assert phone.shape == (4, 4)


def test_v7_hierarchical_training_records_independent_objectives():
    _, training = train_hierarchical_v7(
        _dataset(),
        epochs=1,
        batch_size=8,
        learning_rate=0.001,
        seed=7,
        coarse_loss_mode="none",
        fine_loss_mode="sqrt_inverse",
        coarse_loss_coefficient=2.0,
        fine_loss_coefficient=1.0,
        hard_negative_centers={(0, 2)},
        hard_negative_weight=2.0,
    )

    assert training["coarse_loss_mode"] == "none"
    assert training["fine_loss_mode"] == "sqrt_inverse"
    assert training["coarse_loss_coefficient"] == 2.0
    assert training["hard_negative_count"] == 1
