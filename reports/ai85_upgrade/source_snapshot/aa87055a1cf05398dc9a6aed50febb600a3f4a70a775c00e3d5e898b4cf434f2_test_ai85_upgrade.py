"""Regression checks for checkpoint continuation, subject isolation and fusion."""

import numpy as np
import pytest
import torch
import yaml

from training.dmd.ai85_action import (
    AttentionResNetGRUClassifier,
    finetune,
    load_inherited_state,
    negative_evidence_fusion,
)
from training.dmd.holdout_guard import HoldoutAccessError
from training.dmd.run_ai85_upgrade import DEFAULT_CONFIG, validate_pool
from training.dmd.v5_resnet_gru import EmbeddingClipDataset, SubjectEmbeddings
from training.dmd.v6_action_hierarchy import (
    HierarchicalResNetGRUClassifier,
    hierarchical_probabilities,
)


def test_uniform_attention_preserves_inherited_outputs_and_receives_gradients():
    torch.manual_seed(11)
    base = HierarchicalResNetGRUClassifier(embedding_dim=8, hidden_size=4).eval()
    model = AttentionResNetGRUClassifier(embedding_dim=8, hidden_size=4).eval()
    load_inherited_state(model, base.state_dict())
    sample = torch.randn(3, 9, 8)
    expected = base(sample)
    actual = model(sample)
    for a, b in zip(expected, actual, strict=True):
        torch.testing.assert_close(a, b, atol=2e-7, rtol=2e-6)
    probabilities = hierarchical_probabilities(*actual)
    torch.testing.assert_close(probabilities.sum(1), torch.ones(3))
    (-probabilities[:, 0].log().mean()).backward()
    assert model.temporal_attention.weight.grad.abs().sum() > 0


def test_attention_normalizes_across_time_with_variable_clip_length():
    model = AttentionResNetGRUClassifier(embedding_dim=8, hidden_size=4)
    encoded = torch.randn(2, 17, 8)
    weights = model.attention_weights(encoded)
    torch.testing.assert_close(weights.sum(1), torch.ones(2, 1))
    coarse, phone = model(torch.randn(2, 17, 8))
    assert coarse.shape == (2, 3) and phone.shape == (2, 4)


def test_checkpoint_migration_rejects_missing_encoder_weights():
    base = HierarchicalResNetGRUClassifier(embedding_dim=8, hidden_size=4)
    state = base.state_dict()
    del state["gru.weight_ih_l0"]
    with pytest.raises(ValueError, match="incompatible inherited checkpoint"):
        load_inherited_state(AttentionResNetGRUClassifier(embedding_dim=8, hidden_size=4), state)


@pytest.mark.parametrize("subject", ["gZ-37", "gE-28", "ge_28"])
def test_continuation_blocks_consumed_holdouts(subject):
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    with pytest.raises(HoldoutAccessError):
        validate_pool(config, subject)


def test_continuation_changes_model_parameters_on_supervised_training_data():
    rng = np.random.default_rng(7)
    subject = SubjectEmbeddings(
        subject_id="synthetic-training-only", embeddings=rng.normal(size=(18, 8)).astype("f4"),
        timestamps=np.arange(18) * 0.5, labels=np.asarray([0, 1, 2, 3, 4, 5] * 3),
    )
    dataset = EmbeddingClipDataset([subject], sequence_length=3, stride=1, labeled_only=True)
    model = AttentionResNetGRUClassifier(embedding_dim=8, hidden_size=4)
    before = model.temporal_attention.weight.detach().clone()
    metadata = finetune(model, dataset, epochs=1, batch_size=8, learning_rate=0.001, seed=8)
    assert not torch.equal(before, model.temporal_attention.weight)
    assert metadata["samples"] == len(dataset)
    assert metadata["seed"] == 8
    assert metadata["history"][0]["loss"] > 0


def test_negative_fusion_preserves_missing_evidence_and_phone_supported_events():
    events = [{"start_seconds": 1.0, "end_seconds": 2.0},
              {"start_seconds": 4.0, "end_seconds": 5.0},
              {"start_seconds": 10.0, "end_seconds": 11.0}]
    timestamps = np.asarray([1, 2, 4, 5])
    probabilities = np.asarray([[.8, .05, .05, .05, .05, 0],
                                [.9, .025, .025, .025, .025, 0],
                                [.1, .8, .05, .025, .025, 0],
                                [.1, .8, .05, .025, .025, 0]])
    original = [dict(e) for e in events]
    kept, audit = negative_evidence_fusion(
        events, timestamps, probabilities, minimum_safe_probability=.5, minimum_fraction=.5,
    )
    assert kept == [events[1], events[2]]
    assert events == original
    assert [r["vetoed"] for r in audit] == [True, False, False]
    assert audit[2]["strong_safe_fraction"] is None


def test_error_audit_accounts_for_duplicate_false_positives():
    from training.dmd.ai85_audit import read_json

    audit = read_json("reports/ai85_upgrade/error_analysis.json")
    metrics = audit["recomputed_metrics"]
    cases = audit["false_positive_cases"]
    assert len(cases) == metrics["false_positives"]
    assert sum(c["kind"] == "DUPLICATE_MATCH" for c in cases) == metrics["duplicate_events"]
    assert sum(c["result"] == "FN" for c in audit["cases"]) == metrics["false_negatives"]
