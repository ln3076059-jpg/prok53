"""Verify V11 deliverables, governance invariants, and prediction/source hashes."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v11_research"
REPORTS = [
    "01_LITERATURE_REVIEW.md",
    "02_REPOSITORY_AUDIT.md",
    "03_DATA_GOVERNANCE.md",
    "04_FROZEN_BASELINE.md",
    "05_FALSE_NEGATIVE_ROOT_CAUSE.md",
    "06_BEHAVIOR_ROI_EXPERIMENTS.md",
    "07_CAUSAL_MOTION_EXPERIMENTS.md",
    "08_CLIP_AND_DUALSCOPE_COMPARISON.md",
    "09_SUBJECT_GENERALIZATION.md",
    "10_DRIVER_PHONE_ASSOCIATION.md",
    "11_FUSION_ABLATION.md",
    "12_TEMPORAL_LOCALIZATION.md",
    "13_SEATBELT_EXPERIMENTS.md",
    "14_EVENT_METRICS.md",
    "15_RUNTIME_RESULTS.md",
    "16_TEST_RESULTS.md",
    "17_FINAL_AI_RESEARCH_REVIEW.md",
]
MACHINE_ARTIFACTS = [
    "repository_environment.json",
    "dataset_manifest.json",
    "model_manifest.json",
    "literature_review.json",
    "fn_root_cause.json",
    "representation_experiments.json",
    "generalization_experiments.json",
    "experiment_registry.json",
    "metric_comparison.json",
    "confusion_matrices.json",
    "per_subject_results.json",
    "event_match_evidence.json",
    "test_results.json",
]


def _read(output: Path, name: str) -> Any:
    return json.loads((output / name).read_text(encoding="utf-8"))


def verify(output: Path = DEFAULT_OUTPUT, require_test_report: bool = True) -> dict[str, Any]:
    reports = REPORTS if require_test_report else [name for name in REPORTS if name != "16_TEST_RESULTS.md"]
    required = reports + MACHINE_ARTIFACTS
    if not require_test_report:
        required.remove("test_results.json")
    missing = [name for name in required if not (output / name).is_file()]
    undersized = [name for name in reports if (output / name).is_file() and (output / name).stat().st_size < 500]
    if missing or undersized:
        raise ValueError(f"Deliverable validation failed: missing={missing}, undersized={undersized}")

    dataset = _read(output, "dataset_manifest.json")
    fn = _read(output, "fn_root_cause.json")
    reps = _read(output, "representation_experiments.json")
    gen = _read(output, "generalization_experiments.json")
    registry = _read(output, "experiment_registry.json")
    metrics = _read(output, "metric_comparison.json")
    models = _read(output, "model_manifest.json")
    if dataset["rights_accepted"] or dataset["canonical_eligible"]:
        raise ValueError("Dataset rights invariant changed")
    if dataset["protected_holdouts_accessed_in_v11"]:
        raise ValueError("Protected holdout access was reported")
    if set(dataset["development_subjects"]) & set(dataset["protected_holdouts"]):
        raise ValueError("Development and protected subjects overlap")
    if fn["summary"]["cases"] != 13 or fn["summary"]["valid_auxiliary_box"] != 12:
        raise ValueError("Frozen 13-FN population changed")
    if fn["recovered_events_claimed"]:
        raise ValueError("FN diagnostic claims event recovery")
    if reps["production_approved"] or gen["production_approved"] or metrics["production_approved"]:
        raise ValueError("Development artifact claims production approval")
    if registry["production_approved"] or registry["final_status"] != "RESEARCH_ONLY":
        raise ValueError("Registry final decision is invalid")
    if models["v10_hash_reconciliation"]["status"] != "PASS":
        raise ValueError("Checkpoint reconciliation failed")
    if any(row["load_executed"] for row in models["checkpoints"]):
        raise ValueError("Checkpoint inventory reports deserialization")

    prediction_count = 0
    for family in (reps, gen):
        for representation in family["representations"]:
            for fold in representation["folds"]:
                relative = fold["prediction_path"]
                if any(subject in relative for subject in dataset["protected_holdouts"]):
                    raise ValueError(f"Protected subject in prediction path: {relative}")
                if sha256_file(ROOT / relative) != fold["prediction_sha256"]:
                    raise ValueError(f"Prediction hash mismatch: {relative}")
                prediction_count += 1
    for relative, expected in registry["source_hashes"].items():
        if sha256_file(ROOT / relative) != expected:
            raise ValueError(f"Source artifact changed: {relative}")

    manifest = {}
    for path in sorted(output.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "validation.json"}:
            manifest[path.relative_to(ROOT).as_posix()] = {
                "bytes": path.stat().st_size,
                "sha256": sha256_file(path),
            }
    result = {
        "status": "PASS",
        "reports": len(reports),
        "artifacts_hashed": len(manifest),
        "prediction_artifacts": prediction_count,
        "checkpoints_inventoried": models["checkpoint_count"],
        "protected_holdouts_accessed": False,
        "production_approved": False,
        "final_status": "RESEARCH_ONLY",
    }
    (output / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "validation.json").write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--allow-missing-test-report", action="store_true")
    args = parser.parse_args()
    print(json.dumps(verify(args.output, not args.allow_missing_test_report), indent=2))


if __name__ == "__main__":
    main()
