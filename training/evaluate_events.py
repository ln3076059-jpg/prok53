from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

REQUIRED = {"video_id", "event_type", "start_seconds", "end_seconds"}


def _read(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        rows = list(csv.DictReader(handle))
    missing = REQUIRED - set(rows[0] if rows else [])
    if missing:
        raise ValueError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def _overlap(a: dict, b: dict, tolerance: float) -> bool:
    if a["video_id"] != b["video_id"] or a["event_type"] != b["event_type"]:
        return False
    a_start, a_end = float(a["start_seconds"]), float(a["end_seconds"])
    b_start, b_end = float(b["start_seconds"]), float(b["end_seconds"])
    return b_start <= a_end + tolerance and a_start <= b_end + tolerance


def evaluate(
    truth_rows: list[dict],
    prediction_rows: list[dict],
    video_minutes: float,
    tolerance_seconds: float = 0.5,
) -> dict:
    matched_truth: set[int] = set()
    matches: list[tuple[int, int]] = []
    duplicates = 0
    for prediction_index, prediction in enumerate(prediction_rows):
        candidates = [
            truth_index
            for truth_index, truth in enumerate(truth_rows)
            if _overlap(truth, prediction, tolerance_seconds)
        ]
        unmatched = [index for index in candidates if index not in matched_truth]
        if unmatched:
            truth_index = min(
                unmatched,
                key=lambda index: abs(
                    float(truth_rows[index]["start_seconds"]) - float(prediction["start_seconds"])
                ),
            )
            matched_truth.add(truth_index)
            matches.append((truth_index, prediction_index))
        elif candidates:
            duplicates += 1

    report = {"status": "MEASURED", "event_types": {}}
    event_types = sorted({row["event_type"] for row in truth_rows + prediction_rows})
    for event_type in event_types:
        truth_indices = {i for i, row in enumerate(truth_rows) if row["event_type"] == event_type}
        prediction_indices = {
            i for i, row in enumerate(prediction_rows) if row["event_type"] == event_type
        }
        type_matches = [
            (t, p) for t, p in matches if t in truth_indices and p in prediction_indices
        ]
        tp = len(type_matches)
        fp = len(prediction_indices) - tp
        fn = len(truth_indices) - tp
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        delays = [
            float(prediction_rows[p]["start_seconds"]) - float(truth_rows[t]["start_seconds"])
            for t, p in type_matches
        ]
        report["event_types"][event_type] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "true_positives": tp,
            "false_positives": fp,
            "missed_events": fn,
            "false_events_per_minute": fp / video_minutes if video_minutes > 0 else None,
            "mean_time_to_detection_seconds": sum(delays) / len(delays) if delays else None,
        }
    report["duplicate_events"] = duplicates
    report["video_minutes"] = video_minutes
    report["matching_tolerance_seconds"] = tolerance_seconds
    return report


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate event-level predictions against frozen human ground truth"
    )
    parser.add_argument("truth", type=Path, nargs="?")
    parser.add_argument("predictions", type=Path, nargs="?")
    parser.add_argument("--video-minutes", type=float, default=0.0)
    parser.add_argument("--tolerance-seconds", type=float, default=0.5)
    parser.add_argument("--output", type=Path, default=Path("reports/event_evaluation.json"))
    args = parser.parse_args()
    if not args.truth or not args.predictions:
        report = {
            "status": "NOT_RUN",
            "reason": "human ground truth and predictions are required",
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
        print(json.dumps(report, indent=2))
        return
    report = evaluate(
        _read(args.truth),
        _read(args.predictions),
        args.video_minutes,
        args.tolerance_seconds,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
