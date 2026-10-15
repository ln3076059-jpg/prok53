import hashlib
import json
from pathlib import Path

RESULT_PATH = Path("reports/v5_research_upgrade/v5_experiment_results.json")
ROUND2_RESULT_PATH = Path(
    "reports/v5_research_upgrade/v5_experiment_results_round2.json"
)


def _result():
    return json.loads(RESULT_PATH.read_text(encoding="utf-8"))


def _round2_result():
    return json.loads(ROUND2_RESULT_PATH.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v5_loso_is_subject_disjoint_and_does_not_access_holdouts():
    result = _result()
    development = set(result["development_subjects"])

    assert result["holdouts_accessed"] == []
    assert set(result["forbidden_consumed_holdouts"]) == {"gZ-37", "gE-28"}
    for fold in result["folds"]:
        train = set(fold["training_subjects"])
        heldout = fold["heldout_subject"]
        assert heldout not in train
        assert train | {heldout} == development


def test_v5_pooled_event_counts_and_rates_reconcile():
    result = _result()
    for experiment in result["experiments"]:
        metrics = experiment["pooled_metrics"]
        tp = metrics["true_positives"]
        fp = metrics["false_positives"]
        fn = metrics["false_negatives"]
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

        assert abs(metrics["precision"] - precision) <= 0.0005
        assert abs(metrics["recall"] - recall) <= 0.0005
        assert abs(metrics["f1"] - f1) <= 0.0005
        assert tp + fn == 29


def test_v5_clip_confusion_matrix_reconciles_to_evaluated_clips():
    metrics = _result()["clip_metrics"]
    matrix_total = sum(sum(row) for row in metrics["confusion_matrix"])
    support_total = sum(row["support"] for row in metrics["per_class"].values())

    assert matrix_total == metrics["evaluated_clips"] == support_total


def test_v5_selection_obeys_locked_rule_and_safety_governance():
    result = _result()
    eligible = [
        row
        for row in result["experiments"]
        if row["pooled_metrics"]["false_alarms_per_minute"] <= 0.30
    ]
    expected = max(
        eligible,
        key=lambda row: (
            row["pooled_metrics"]["f1"],
            row["pooled_metrics"]["recall"],
            -row["pooled_metrics"]["false_alarms_per_minute"],
        ),
    )

    assert result["selected_candidate"] == expected["experiment_id"]
    assert result["governance"]["action_only_event_status"] == "NEEDS_REVIEW"
    assert result["governance"]["automatic_violation"] is False
    assert result["governance"]["generalization_claim"] == "NOT_ESTABLISHED"


def test_v5_round2_loso_integrity_and_clip_totals():
    result = _round2_result()
    development = set(result["development_subjects"])

    assert result["experiment_round"] == 2
    assert result["holdouts_accessed"] == []
    assert set(result["forbidden_consumed_holdouts"]) == {"gZ-37", "gE-28"}
    for fold in result["folds_b3"]:
        train = set(fold["training_subjects"])
        heldout = fold["heldout_subject"]
        assert heldout not in train
        assert train | {heldout} == development

    metrics = result["clip_metrics_b3"]
    assert sum(sum(row) for row in metrics["confusion_matrix"]) == 2604
    assert sum(row["support"] for row in metrics["per_class"].values()) == 2604
    assert metrics["evaluated_clips"] == 2604


def test_v5_round2_metrics_selection_and_recovery_reconcile():
    result = _round2_result()
    for experiment in result["experiments"]:
        metrics = experiment["pooled_metrics"]
        assert metrics["true_positives"] + metrics["false_negatives"] == 29

    eligible = [
        row
        for row in result["experiments"]
        if row["pooled_metrics"]["false_alarms_per_minute"] <= 0.30
    ]
    expected = max(
        eligible,
        key=lambda row: (
            row["pooled_metrics"]["f1"],
            row["pooled_metrics"]["recall"],
            -row["pooled_metrics"]["false_alarms_per_minute"],
        ),
    )
    assert result["selected_candidate"] == expected["experiment_id"]
    assert result["selected_candidate"] == "B2_TINY_CNN_TEMPORAL"
    assert result["recovered_count"] == len(result["v4_false_negatives_recovered"])
    for experiment_id, count in result["recovered_count_by_experiment"].items():
        assert count == len(result["v4_false_negative_recovery_by_experiment"][experiment_id])


def test_v5_round2_checkpoint_hashes_and_metadata_are_inspectable():
    result = _round2_result()
    metadata_path = Path(result["checkpoint_metadata_path_b3"])
    assert metadata_path.is_file()
    assert _sha256(metadata_path) == result["checkpoint_metadata_sha256_b3"]

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["activation_state"] == "DEVELOPMENT_ONLY_NOT_VALIDATED"
    assert metadata["automatic_violation"] is False
    for fold in result["folds_b3"]:
        checkpoint_path = Path(fold["checkpoint_path"])
        checkpoint_sha256 = fold["checkpoint_sha256"]
        assert len(checkpoint_sha256) == 64
        assert all(character in "0123456789abcdef" for character in checkpoint_sha256)
        # Checkpoints are intentionally not distributed while DMD rights remain
        # pending. Verify the artifact byte-for-byte when it exists locally.
        if checkpoint_path.is_file():
            assert _sha256(checkpoint_path) == checkpoint_sha256
