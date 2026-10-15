"""Validate V6 metrics, provenance, artifacts, reports, and governance gates."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "v6_completion"
REQUIRED_REPORTS = [
    "01_REQUIREMENTS_TRACEABILITY.md",
    "02_CURRENT_REPOSITORY_AUDIT.md",
    "03_DATA_AND_ARTIFACT_INVENTORY.md",
    "04_CLASS_COLLAPSE_ROOT_CAUSE.md",
    "05_SUBJECT_GENERALIZATION_ANALYSIS.md",
    "06_PHONE_EXPERIMENT_PLAN.md",
    "07_PHONE_ABLATION_RESULTS.md",
    "08_SEATBELT_IMPROVEMENT.md",
    "09_METRICS_COMPARISON.md",
    "10_END_TO_END_TEST_RESULTS.md",
    "11_PERFORMANCE_BENCHMARK.md",
    "12_SCIENTIFIC_VALIDITY.md",
    "13_REMAINING_LIMITATIONS.md",
    "14_FINAL_COMPLETION_REPORT.md",
    "references.bib",
]


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    errors: list[str] = []
    checks: dict[str, Any] = {}
    v6 = _read_json(REPORT_DIR / "v6_class_collapse_ablation.json")
    v5 = _read_json(ROOT / "reports/v5_research_upgrade/v5_experiment_results_round2.json")
    seatbelt = _read_json(REPORT_DIR / "v6_seatbelt_evaluation.json")
    performance = _read_json(REPORT_DIR / "v6_runtime_benchmark.json")

    missing_reports = [
        name
        for name in REQUIRED_REPORTS
        if not (REPORT_DIR / name).is_file() or (REPORT_DIR / name).stat().st_size < 100
    ]
    checks["required_reports"] = {"expected": 15, "missing_or_empty": missing_reports}
    if missing_reports:
        errors.append(f"missing or empty reports: {missing_reports}")

    folds = [fold for variant in v6["variants"] for fold in variant["folds"]]
    checks["ablation_folds"] = len(folds)
    if len(folds) != 12:
        errors.append(f"expected 12 LOSO folds, found {len(folds)}")
    if v6["holdouts_accessed"]:
        errors.append(f"consumed holdouts were accessed: {v6['holdouts_accessed']}")
    for fold in folds:
        checkpoint = ROOT / fold["checkpoint_path"]
        if not checkpoint.is_file():
            errors.append(f"missing checkpoint: {checkpoint}")
        elif _sha256(checkpoint) != fold["checkpoint_sha256"]:
            errors.append(f"checkpoint hash mismatch: {checkpoint}")

    expected_baselines = {
        "B0_V4_CURRENT": (3, 5, 26, 0.1620),
        "B2_TINY_CNN_TEMPORAL": (6, 5, 23, 0.3000),
        "B3_RESNET18_GRU": (10, 7, 19, 0.4348),
        "B8_B3_LATE_FUSION": (11, 7, 18, 0.4681),
    }
    for experiment_id, expected in expected_baselines.items():
        metrics = next(
            row["pooled_metrics"]
            for row in v5["experiments"]
            if row["experiment_id"] == experiment_id
        )
        observed = (
            metrics["true_positives"],
            metrics["false_positives"],
            metrics["false_negatives"],
            metrics["f1"],
        )
        if observed != expected:
            errors.append(f"baseline mismatch for {experiment_id}: {observed} != {expected}")
    checks["historical_baselines"] = "MATCHED"

    for row in v6["selection_rows"]:
        expected_eligible = all(row["gates"].values())
        if row["eligible"] != expected_eligible:
            errors.append(f"selection eligibility mismatch: {row['experiment_id']}")
    expected_selected = [row for row in v6["selection_rows"] if row["eligible"]]
    if bool(v6["selected_candidate"]) != bool(expected_selected):
        errors.append("selected candidate does not match eligible rows")
    checks["eligible_candidate_count"] = len(expected_selected)

    raw_support = sum(
        row["support"] for row in seatbelt["raw"]["per_class"].values()
    )
    raw_confusion = sum(
        sum(row.values()) for row in seatbelt["raw"]["confusion_matrix"].values()
    )
    if raw_support != 641 or raw_confusion != 641 or seatbelt["sample_count"] != 641:
        errors.append("seatbelt sample totals are inconsistent")
    checks["seatbelt_samples"] = seatbelt["sample_count"]

    if performance["full_backend_pipeline"]["frames"] <= 0:
        errors.append("full backend benchmark has no frames")
    checks["full_pipeline_fps"] = performance["full_backend_pipeline"]["processing_fps"]

    serialized_governance = json.dumps(v6["governance"]).upper()
    if "PRODUCTION_APPROVED\": TRUE" in serialized_governance:
        errors.append("V6 governance unexpectedly marks production approved")
    checks["governance"] = v6["governance"]

    result = {
        "status": "PASS" if not errors else "FAIL",
        "checks": checks,
        "errors": errors,
    }
    output = REPORT_DIR / "v6_validation.json"
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))
    if errors:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
