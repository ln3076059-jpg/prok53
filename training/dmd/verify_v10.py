"""Verify V10 research artifacts, governance invariants, and source hashes."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v10_research"
REPORTS = {
    1: "01_REPOSITORY_AND_ENVIRONMENT_AUDIT.md",
    2: "02_PAPER_REVIEW_AND_REPRODUCIBILITY.md",
    3: "03_DATASET_AND_LICENSE_AUDIT.md",
    4: "04_MODEL_CHECKPOINT_INVENTORY.md",
    5: "05_DMD_FN_FP_ROOT_CAUSE.md",
    6: "06_PHONE_DETECTION_AND_ASSOCIATION.md",
    7: "07_BEHAVIOR_CENTRIC_REPRESENTATION.md",
    8: "08_MOTION_AND_VIDEO_ACTION_ABLATIONS.md",
    9: "09_CLASS_IMBALANCE_AND_SAFE_DRIVING.md",
    10: "10_MULTI_VIEW_FUSION_EXPERIMENTS.md",
    11: "11_TEMPORAL_EVENT_EXPERIMENTS.md",
    12: "12_SEATBELT_RESEARCH_RESULTS.md",
    13: "13_METRICS_COMPARISON.md",
    14: "14_GENERALIZATION_VALIDATION.md",
    15: "15_RUNTIME_BENCHMARK.md",
    16: "16_TEST_RESULTS.md",
    17: "17_FINAL_AI_LEAD_REVIEW.md",
}


def _read(output: Path, name: str) -> Any:
    return json.loads((output / name).read_text(encoding="utf-8"))


def verify(output: Path = DEFAULT_OUTPUT, require_test_report: bool = True) -> dict[str, Any]:
    required = list(REPORTS.values())
    if not require_test_report:
        required.remove(REPORTS[16])
    missing = [name for name in required if not (output / name).is_file()]
    undersized = [
        name
        for name in required
        if (output / name).is_file() and (output / name).stat().st_size < 500
    ]
    if missing or undersized:
        raise ValueError(f"Report validation failed: missing={missing}, undersized={undersized}")

    dataset = _read(output, "dataset_manifest.json")
    representation = _read(output, "representation_probe.json")
    root_cause = _read(output, "fn_root_cause.json")
    metrics = _read(output, "metrics_comparison.json")
    models = _read(output, "model_manifest.json")
    registry = _read(output, "experiment_registry.json")
    if dataset["rights_accepted"] or dataset["canonical_eligible"]:
        raise ValueError("V10 dataset rights invariant changed")
    if dataset["protected_holdouts_accessed_in_v10"]:
        raise ValueError("Protected holdout access was reported")
    if set(dataset["development_subjects"]) & set(dataset["protected_holdouts"]):
        raise ValueError("Development and protected subject lists overlap")
    if representation["production_approved"] or metrics["production_approved"]:
        raise ValueError("A development artifact claims production approval")
    if root_cause["summary"]["cases"] != 13 or root_cause["summary"]["valid_proposal"] != 12:
        raise ValueError("Frozen phone FN population changed")
    if any(item["load_executed"] for item in models["checkpoints"]):
        raise ValueError("Checkpoint inventory reports unsafe deserialization")
    if registry["production_approved"]:
        raise ValueError("Experiment registry claims production approval")

    prediction_count = 0
    for item in representation["representations"]:
        for fold in item["folds"]:
            relative = fold["prediction_path"]
            if any(subject in relative for subject in dataset["protected_holdouts"]):
                raise ValueError(f"Protected subject appears in prediction path: {relative}")
            path = ROOT / relative
            if sha256_file(path) != fold["prediction_sha256"]:
                raise ValueError(f"Prediction hash mismatch: {relative}")
            prediction_count += 1
    for relative, expected in metrics["source_hashes"].items():
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
        "reports": len(required),
        "artifacts_hashed": len(manifest),
        "prediction_artifacts": prediction_count,
        "checkpoints_inventoried": models["checkpoint_count"],
        "protected_holdouts_accessed": False,
        "production_approved": False,
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
    print(
        json.dumps(
            verify(args.output, require_test_report=not args.allow_missing_test_report),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
