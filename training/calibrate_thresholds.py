from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from training.common import sha256_file
from training.train_fusion import choose_threshold


def calibrate(
    path: Path,
    output: Path,
    model_path: Path,
    validation_manifest: Path,
    minimum_recall: float = 0.70,
    objective: str = "MAXIMIZE_F1_SUBJECT_TO_MINIMUM_RECALL",
) -> dict:
    for artifact_name, artifact_path in (
        ("model", model_path),
        ("validation manifest", validation_manifest),
    ):
        if not artifact_path.is_file():
            raise ValueError(f"{artifact_name} is missing: {artifact_path}")
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    required = {"split", "class_name", "y_true", "score"}
    missing = required - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"calibration CSV is missing columns: {sorted(missing)}")
    groups: defaultdict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if row["split"].lower() != "val":
            continue
        groups[row["class_name"]].append(row)
    if not groups:
        raise ValueError("calibration requires validation rows")
    report = {
        "schema_version": 2,
        "status": "CALIBRATED_ON_VALIDATION",
        "created_at": datetime.now(UTC).isoformat(),
        "method": "EXHAUSTIVE_OBSERVED_SCORE_THRESHOLD_SEARCH",
        "metric_objective": objective,
        "minimum_recall": minimum_recall,
        "model": {"path": str(model_path), "sha256": sha256_file(model_path)},
        "validation_dataset_manifest": {
            "path": str(validation_manifest),
            "sha256": sha256_file(validation_manifest),
        },
        "score_artifact": {"path": str(path), "sha256": sha256_file(path)},
        "thresholds": {},
        "classes": {},
        "input_split_rows": dict(Counter(row["split"].lower() for row in rows)),
        "test_rows_used": 0,
    }
    for class_name, values in sorted(groups.items()):
        labels = np.asarray([int(item["y_true"]) for item in values], dtype=np.float64)
        scores = np.asarray([float(item["score"]) for item in values], dtype=np.float64)
        if len(np.unique(labels)) != 2:
            raise ValueError(f"{class_name} needs positive and negative validation rows")
        selected = choose_threshold(labels, scores, minimum_recall)
        report["thresholds"][class_name] = selected["threshold"]
        report["classes"][class_name] = {**selected, "validation_rows": len(values)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Calibrate per-class thresholds on validation only"
    )
    parser.add_argument("csv", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--validation-manifest", type=Path, required=True)
    parser.add_argument(
        "--output", type=Path, default=Path("reports/v2_threshold_calibration.json")
    )
    parser.add_argument("--minimum-recall", type=float, default=0.70)
    parser.add_argument("--objective", default="MAXIMIZE_F1_SUBJECT_TO_MINIMUM_RECALL")
    args = parser.parse_args()
    print(
        json.dumps(
            calibrate(
                args.csv,
                args.output,
                args.model,
                args.validation_manifest,
                args.minimum_recall,
                args.objective,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
