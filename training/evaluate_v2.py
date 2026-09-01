from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def classification_report(rows: list[tuple[str, str]], class_names: list[str]) -> dict:
    confusion: defaultdict[str, Counter] = defaultdict(Counter)
    for truth, predicted in rows:
        confusion[truth][predicted] += 1
    classes = {}
    for name in class_names:
        tp = confusion[name][name]
        fp = sum(confusion[truth][name] for truth in class_names if truth != name)
        fn = sum(confusion[name][predicted] for predicted in class_names if predicted != name)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        classes[name] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": sum(confusion[name].values()),
        }
    return {
        "classes": classes,
        "macro_f1": sum(item["f1"] for item in classes.values()) / len(classes),
        "accuracy": sum(confusion[name][name] for name in class_names) / max(len(rows), 1),
        "confusion_matrix": {
            truth: {predicted: confusion[truth][predicted] for predicted in class_names}
            for truth in class_names
        },
    }


def evaluate_classifier(weights: Path, data: Path, split: str, output: Path) -> dict:
    if split == "test" and not Path("reports/model_lock_v2.json").is_file():
        raise ValueError("frozen V2 test guard: lock all component models before test")
    from ultralytics import YOLO

    root = data / split
    class_names = sorted(path.name for path in root.iterdir() if path.is_dir())
    expected = ["seatbelt_fastened", "seatbelt_unfastened", "uncertain_or_occluded"]
    if class_names != sorted(expected):
        raise ValueError(f"classifier class schema mismatch: {class_names}")
    images = [
        (class_name, path)
        for class_name in class_names
        for path in sorted((root / class_name).glob("*"))
        if path.is_file()
    ]
    if not images:
        raise ValueError(f"classifier {split} split is empty")
    model = YOLO(str(weights))
    predictions = model.predict([str(path) for _, path in images], verbose=False)
    rows = []
    for (truth, _), result in zip(images, predictions, strict=True):
        if result.probs is None:
            raise ValueError("classifier produced no probabilities")
        predicted = str(result.names[int(result.probs.top1)])
        rows.append((truth, predicted))
    report = {"task": "classify", "split": split, **classification_report(rows, class_names)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def evaluate_detector(weights: Path, data: Path, split: str, output: Path) -> dict:
    if split == "test" and not Path("reports/model_lock_v2.json").is_file():
        raise ValueError("frozen V2 test guard: lock all component models before test")
    from ultralytics import YOLO

    result = YOLO(str(weights)).val(
        data=str(data), split=split, plots=True, project=str(output.parent), name=output.stem
    )
    names = result.names
    classes = {}
    for index in range(len(result.box.p)):
        precision = float(result.box.p[index])
        recall = float(result.box.r[index])
        classes[str(names[index])] = {
            "precision": precision,
            "recall": recall,
            "f1": 0.0
            if precision + recall == 0
            else 2 * precision * recall / (precision + recall),
            "ap50": float(result.box.ap50[index]),
            "ap50_95": float(result.box.ap[index]),
        }
    report = {
        "task": "detect",
        "split": split,
        "classes": classes,
        "mAP50": float(result.box.map50),
        "mAP50_95": float(result.box.map),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Per-component V2 evaluation")
    parser.add_argument("weights", type=Path)
    parser.add_argument("--task", choices=["detect", "classify"], required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--output", type=Path, default=Path("reports/v2_evaluation.json"))
    args = parser.parse_args()
    function = evaluate_detector if args.task == "detect" else evaluate_classifier
    print(json.dumps(function(args.weights, args.data, args.split, args.output), indent=2))


if __name__ == "__main__":
    main()
