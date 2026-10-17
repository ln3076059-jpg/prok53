"""Verify V12 reports, provenance, measurements, and governance invariants."""

from __future__ import annotations

import json
from pathlib import Path

from training.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v12_research"
REPORTS = [f"{i:02d}_{name}.md" for i, name in enumerate((
    "CODE_AND_DATA_AUDIT", "METRIC_RECONCILIATION", "ONLINE_CANDIDATE",
    "PHONE_ROI_ASSOCIATION", "REPRESENTATION_ABLATIONS", "EVENT_REPLAY",
    "SEATBELT_RESULTS", "RUNTIME_AND_TESTS", "FINAL_RESEARCH_REPORT"), start=1)]
JSONS = ["experiment_plan.json", "experiment_registry.json", "dataset_manifest.json",
         "model_manifest.json", "metric_comparison.json", "metric_reconciliation.json",
         "streaming_audit.json", "roi_audit.json", "event_match_records.json",
         "test_results.json", "diagnostic_execution.json", "repository_environment.json"]


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def verify() -> dict:
    missing = [name for name in REPORTS + JSONS if not (OUT / name).is_file()]
    small = [
        name
        for name in REPORTS
        if (OUT / name).is_file() and (OUT / name).stat().st_size < 500
    ]
    if missing or small:
        raise ValueError(f"missing={missing}, undersized={small}")
    dataset, reconcile = read("dataset_manifest.json"), read("metric_reconciliation.json")
    streaming, roi = read("streaming_audit.json"), read("roi_audit.json")
    replay, metrics = read("event_match_records.json"), read("metric_comparison.json")
    registry, tests = read("experiment_registry.json"), read("test_results.json")
    if dataset["permissions"].get("RIGHTS_ACCEPTED") != "false":
        raise ValueError("rights invariant changed")
    if dataset["protected_media_or_labels_opened"]:
        raise ValueError("protected holdout access reported")
    if set(dataset["denied_operations"]) != {"linear_head", "calibration", "fusion",
                                                  "boundary", "backbone", "upload"}:
        raise ValueError("fitting/upload permission gates incomplete")
    if reconcile["status"] != "PASS" or reconcile["discrepancies"]:
        raise ValueError("metric reconciliation failed")
    for relative, digest in reconcile["source_hashes"].items():
        if sha256_file(ROOT / relative) != digest:
            raise ValueError(f"source hash mismatch: {relative}")
    if len(streaming["subjects"]) != 3 or not all(r["parity"] for r in streaming["subjects"]):
        raise ValueError("stream parity failed")
    for row in streaming["subjects"]:
        if sha256_file(ROOT / row["source_path"]) != row["source_sha256"]:
            raise ValueError("stream source changed")
    valid_phone_crops = sum(
        row["phone_contained_in_real_pixel_crop"] is True for row in roi["cases"]
    )
    if roi["decoded_frames"] != 13 or valid_phone_crops != 12:
        raise ValueError("ROI evidence population changed")
    if roi["detector_rerun"] or roi["new_event_recovery_claimed"]:
        raise ValueError("ROI diagnostic overclaims")
    for row in roi["cases"]:
        if sha256_file(ROOT / row["roi_path"]) != row["roi_sha256"]:
            raise ValueError("ROI hash mismatch")
    for subject in replay["subjects"]:
        for row in subject["variants"].values():
            counts = row["recomputed_counts"]
            if (counts["tp"], counts["fp"], counts["fn"]) != (
                    row["true_positives"], row["false_positives"], row["false_negatives"]):
                raise ValueError("event count mismatch")
    if metrics["v12_e3_event_metrics"] is not None or metrics["production_approved"]:
        raise ValueError("unmeasured result or production claim")
    if registry["training_performed"] or registry["final_status"] != "RESEARCH_ONLY":
        raise ValueError("registry governance failed")
    for suite in (tests["targeted"], tests["full"]):
        if suite["failures"] or suite["errors"]:
            raise ValueError("test suite failed")
    if tests["frontend"] != "PASS" or tests["backend"] != "PASS":
        raise ValueError("build or backend smoke failed")
    manifest = {}
    for path in sorted(OUT.rglob("*")):
        if path.is_file() and path.name not in {"artifact_manifest.json", "validation.json"}:
            manifest[path.relative_to(ROOT).as_posix()] = {"bytes": path.stat().st_size,
                                                           "sha256": sha256_file(path)}
    result = {"status": "PASS", "reports": len(REPORTS), "artifacts_hashed": len(manifest),
              "prediction_artifacts_reconciled": 30, "original_frames_decoded": 13,
              "training_performed": False, "protected_holdouts_accessed": False,
              "production_approved": False, "final_status": "RESEARCH_ONLY"}
    (OUT / "artifact_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    (OUT / "validation.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
