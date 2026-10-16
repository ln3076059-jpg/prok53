import hashlib
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "v10_research"
PROTECTED = {"gZ-37", "gE-28"}


def _read(name: str) -> dict:
    return json.loads((OUTPUT / name).read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def test_v10_dataset_governance_blocks_training_and_holdout_access():
    manifest = _read("dataset_manifest.json")
    assert manifest["rights_accepted"] is False
    assert manifest["canonical_eligible"] is False
    assert manifest["training_allowed_by_v10_audit"] is False
    assert manifest["external_upload_allowed"] is False
    assert manifest["protected_holdouts_accessed_in_v10"] is False
    assert set(manifest["protected_holdouts"]) == PROTECTED
    assert not (set(manifest["development_subjects"]) & PROTECTED)
    assert manifest["generalization_status"] == "DATA_GENERALIZATION_BLOCKED"


def test_v10_representation_probe_is_reproducible_and_not_promoted():
    result = _read("representation_probe.json")
    assert result["protected_holdouts_accessed"] is False
    assert result["production_approved"] is False
    assert result["hypothesis_outcome"] == "PARTIALLY_SUPPORTED"
    assert result["model_selection_decision"] == "REJECT_FOR_PROMOTION"
    assert result["selected_candidate"] == "D2_TEMPORAL_RESIDUAL"
    assert len(result["representations"]) == 4
    assert sum(fold["samples"] for fold in result["representations"][0]["folds"]) == 2625
    for representation in result["representations"]:
        assert {fold["heldout_subject"] for fold in representation["folds"]} == {
            "gB-9",
            "gC-14",
            "gZ-36",
        }
        for fold in representation["folds"]:
            path = ROOT / fold["prediction_path"]
            assert _sha256(path) == fold["prediction_sha256"]
            with np.load(path, allow_pickle=False) as predictions:
                assert len(predictions["labels"]) == fold["samples"]
                assert predictions["labels"].shape == predictions["predictions"].shape


def test_v10_phone_root_cause_keeps_candidate_yield_separate_from_event_recall():
    result = _read("fn_root_cause.json")
    assert result["summary"]["cases"] == 13
    assert result["summary"]["visible_phone"] == 13
    assert result["summary"]["valid_proposal"] == 12
    assert result["summary"]["verified_occupant_tracks"] == 0
    assert result["summary"]["T1_reaches_action_gate"] == 4
    assert result["detector_candidate_yield_is_event_recall"] is False
    assert result["ground_truth_changed"] is False
    assert result["event_matching_policy_changed"] is False
    hard = next(case for case in result["cases"] if case["event_id"] == "gB-9-phone-06")
    assert hard["phone_visibility"] == "PHONE_VISIBLE"
    assert hard["proposal_available"] is False
    assert "DETECTOR_MISS" in hard["likely_failure_stages"]


def test_v10_model_inventory_does_not_load_or_activate_checkpoints():
    manifest = _read("model_manifest.json")
    assert manifest["checkpoint_count"] >= 1
    assert manifest["safety"] == {
        "pickle_load_performed": False,
        "checkpoint_download_performed": False,
        "production_activation_changed": False,
    }
    assert all(not item["load_executed"] for item in manifest["checkpoints"])
    assert all(len(item["sha256"]) == 64 for item in manifest["checkpoints"])


def test_v10_metrics_do_not_claim_targets_or_production():
    metrics = _read("metrics_comparison.json")
    assert metrics["research_targets_achieved"] is False
    assert metrics["production_approved"] is False
    assert metrics["protected_holdouts_accessed"] is False
    controlled = metrics["event_metrics"]["controlled_binary"]
    counts = (
        controlled["true_positives"],
        controlled["false_positives"],
        controlled["false_negatives"],
    )
    assert counts == (
        16,
        4,
        13,
    )


def test_v10_reports_are_present_and_substantive():
    for index in range(1, 18):
        matches = list(OUTPUT.glob(f"{index:02d}_*.md"))
        if index == 16:
            continue
        assert len(matches) == 1
        assert matches[0].stat().st_size >= 500
