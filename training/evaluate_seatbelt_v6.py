"""Independent V6 evaluation of the existing seatbelt classifier predictions."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from collections.abc import Iterable
from datetime import UTC, datetime
from pathlib import Path

CLASSES = ("seatbelt_fastened", "seatbelt_unfastened", "uncertain_or_occluded")
PROBABILITY_COLUMNS = ("prob_fastened", "prob_unfastened", "prob_uncertain")


def _safe_div(numerator: int | float, denominator: int | float) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _metrics(true_labels: list[str], predictions: list[str]) -> dict[str, object]:
    confusion = {
        true_class: {predicted_class: 0 for predicted_class in CLASSES}
        for true_class in CLASSES
    }
    for true_class, predicted_class in zip(true_labels, predictions, strict=True):
        confusion[true_class][predicted_class] += 1

    per_class: dict[str, dict[str, float | int]] = {}
    for class_name in CLASSES:
        tp = confusion[class_name][class_name]
        fp = sum(confusion[other][class_name] for other in CLASSES if other != class_name)
        fn = sum(confusion[class_name][other] for other in CLASSES if other != class_name)
        precision = _safe_div(tp, tp + fp)
        recall = _safe_div(tp, tp + fn)
        f1 = _safe_div(2 * precision * recall, precision + recall)
        per_class[class_name] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "support": sum(confusion[class_name].values()),
        }

    correct = sum(confusion[class_name][class_name] for class_name in CLASSES)
    recalls = [float(per_class[class_name]["recall"]) for class_name in CLASSES]
    f1_scores = [float(per_class[class_name]["f1"]) for class_name in CLASSES]
    return {
        "confusion_matrix": confusion,
        "per_class": per_class,
        "accuracy": round(_safe_div(correct, len(true_labels)), 6),
        "balanced_accuracy": round(sum(recalls) / len(recalls), 6),
        "macro_f1": round(sum(f1_scores) / len(f1_scores), 6),
    }


def expected_calibration_error(
    confidences: Iterable[float], correctness: Iterable[bool], bins: int = 10
) -> float:
    """Return equal-width top-1 ECE over [0, 1]."""
    pairs = list(zip(confidences, correctness, strict=True))
    if not pairs:
        return 0.0
    total = len(pairs)
    ece = 0.0
    for bin_index in range(bins):
        lower = bin_index / bins
        upper = (bin_index + 1) / bins
        bucket = [
            (confidence, correct)
            for confidence, correct in pairs
            if lower <= confidence <= upper and (bin_index == bins - 1 or confidence < upper)
        ]
        if not bucket:
            continue
        accuracy = sum(int(correct) for _, correct in bucket) / len(bucket)
        mean_confidence = sum(confidence for confidence, _ in bucket) / len(bucket)
        ece += (len(bucket) / total) * abs(accuracy - mean_confidence)
    return round(ece, 6)


def evaluate_rows(
    rows: list[dict[str, str]], fastened_threshold: float = 0.50, unfastened_threshold: float = 0.60
) -> dict[str, object]:
    true_labels: list[str] = []
    raw_predictions: list[str] = []
    policy_predictions: list[str] = []
    confidences: list[float] = []
    correctness: list[bool] = []

    for row in rows:
        probabilities = [float(row[column]) for column in PROBABILITY_COLUMNS]
        raw_index = max(range(len(probabilities)), key=probabilities.__getitem__)
        raw_prediction = CLASSES[raw_index]
        policy_prediction = raw_prediction
        if raw_prediction == "seatbelt_fastened" and probabilities[raw_index] < fastened_threshold:
            policy_prediction = "uncertain_or_occluded"
        elif (
            raw_prediction == "seatbelt_unfastened"
            and probabilities[raw_index] < unfastened_threshold
        ):
            policy_prediction = "uncertain_or_occluded"

        true_class = row["true_class"]
        true_labels.append(true_class)
        raw_predictions.append(raw_prediction)
        policy_predictions.append(policy_prediction)
        confidences.append(probabilities[raw_index])
        correctness.append(raw_prediction == true_class)

    true_counts = Counter(true_labels)
    raw_metrics = _metrics(true_labels, raw_predictions)
    policy_metrics = _metrics(true_labels, policy_predictions)
    false_unfastened = sum(
        true_class == "seatbelt_fastened" and predicted == "seatbelt_unfastened"
        for true_class, predicted in zip(true_labels, policy_predictions, strict=True)
    )
    output_uncertain = sum(predicted == "uncertain_or_occluded" for predicted in policy_predictions)
    forced_abstentions = sum(
        raw != "uncertain_or_occluded" and policy == "uncertain_or_occluded"
        for raw, policy in zip(raw_predictions, policy_predictions, strict=True)
    )

    return {
        "sample_count": len(rows),
        "threshold_policy": {
            "fastened_min_confidence": fastened_threshold,
            "unfastened_min_confidence": unfastened_threshold,
        },
        "raw": raw_metrics,
        "post_policy": policy_metrics,
        "top1_ece_10_bin": expected_calibration_error(confidences, correctness),
        "false_unfastened": {
            "count": false_unfastened,
            "rate_among_true_fastened": round(
                _safe_div(false_unfastened, true_counts["seatbelt_fastened"]), 6
            ),
        },
        "abstention": {
            "output_uncertain_count": output_uncertain,
            "output_uncertain_rate": round(_safe_div(output_uncertain, len(rows)), 6),
            "threshold_forced_count": forced_abstentions,
            "threshold_forced_rate": round(_safe_div(forced_abstentions, len(rows)), 6),
        },
        "limitations": [
            "The source is the existing validation prediction file, not an independent holdout.",
            "The uncertain_or_occluded class has only 25 samples.",
            "ECE is descriptive because the same validation split informed the threshold policy.",
        ],
    }


def evaluate_file(source: Path, output: Path) -> dict[str, object]:
    with source.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    result = evaluate_rows(rows)
    result["source"] = source.as_posix()
    result["source_sha256"] = hashlib.sha256(source.read_bytes()).hexdigest()
    result["generated_at_utc"] = datetime.now(UTC).isoformat()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("reports/calibration/seatbelt_classifier_raw_predictions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v6_completion/v6_seatbelt_evaluation.json"),
    )
    args = parser.parse_args()
    result = evaluate_file(args.source, args.output)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
