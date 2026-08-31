from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

CLASS_NAMES = ["phone", "seatbelt_fastened", "seatbelt_unfastened"]


def f1(precision: float, recall: float) -> float:
    return 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)


def evaluate(weights: Path, data: Path, split: str, output: Path) -> dict:
    if split == "test" and not Path("reports/model_lock.json").exists():
        raise ValueError("frozen test guard: reports/model_lock.json must exist before test evaluation")
    from ultralytics import YOLO

    result = YOLO(str(weights)).val(data=str(data), split=split, plots=True, project=str(output.parent), name=output.stem)
    report = {"split": split, "classes": {}, "mAP50": float(result.box.map50), "mAP50_95": float(result.box.map)}
    for index, name in enumerate(CLASS_NAMES):
        precision = float(result.box.p[index])
        recall = float(result.box.r[index])
        report["classes"][name] = {"precision": precision, "recall": recall, "f1": f1(precision, recall), "ap50": float(result.box.ap50[index]), "ap50_95": float(result.box.ap[index])}
    report["macro_f1"] = sum(item["f1"] for item in report["classes"].values()) / 3
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--data", type=Path, default=Path("datasets/governed/mc3_v1/data.yaml"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=Path("reports/validation_metrics.json"))
    args = parser.parse_args()
    data = yaml.safe_load(args.data.read_text())
    if list(data.get("names", {}).values()) != CLASS_NAMES:
        raise ValueError("dataset class names do not match the canonical order")
    print(json.dumps(evaluate(args.weights, args.data, args.split, args.output), indent=2))


if __name__ == "__main__":
    main()
