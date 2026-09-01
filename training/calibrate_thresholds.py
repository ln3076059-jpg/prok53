from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from training.train_fusion import choose_threshold


def calibrate(path: Path, output: Path, minimum_recall: float = 0.70) -> dict:
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
    report = {"thresholds": {}, "classes": {}, "test_rows_used": 0}
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
    parser = argparse.ArgumentParser(description="Calibrate per-class thresholds on validation only")
    parser.add_argument("csv", type=Path)
    parser.add_argument("--output", type=Path, default=Path("reports/v2_thresholds.json"))
    parser.add_argument("--minimum-recall", type=float, default=0.70)
    args = parser.parse_args()
    print(json.dumps(calibrate(args.csv, args.output, args.minimum_recall), indent=2))


if __name__ == "__main__":
    main()
