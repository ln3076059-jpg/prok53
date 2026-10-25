"""Secondary diagnostic for generic PHONE predictions and generic GT episodes.

This does not replace the frozen action-level V2 benchmark.  It answers a
narrower question: how well does a generic PHONE event stream detect continuous
phone-use episodes after adjacent DMD action labels are collapsed by a declared
gap policy?
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable

from training.dmd.audit_generic_event_ceiling import (
    _ground_truth,
    cluster_generic_episodes,
)
from training.dmd.temporal_eval import TemporalMatchingPolicy, match_temporal_events


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _generic_ground_truth(
    metrics: dict[str, Any], gap_seconds: float
) -> list[dict[str, Any]]:
    clusters = cluster_generic_episodes(_ground_truth(metrics), gap_seconds)
    return [
        {
            "start_seconds": cluster[0]["start_seconds"],
            "end_seconds": cluster[-1]["end_seconds"],
            "duration_seconds": (
                float(cluster[-1]["end_seconds"])
                - float(cluster[0]["start_seconds"])
            ),
            "source_action": "GENERIC_PHONE_USE_EPISODE",
            "member_actions": [event["source_action"] for event in cluster],
            "member_action_count": len(cluster),
        }
        for cluster in clusters
    ]


def _generic_predictions(
    predictions: list[dict[str, Any]], gap_seconds: float
) -> list[dict[str, Any]]:
    clusters = cluster_generic_episodes(predictions, gap_seconds)
    return [
        {
            "event_type": cluster[0].get("event_type", "PHONE"),
            "start_seconds": cluster[0]["start_seconds"],
            "end_seconds": cluster[-1]["end_seconds"],
            "duration": round(
                float(cluster[-1]["end_seconds"])
                - float(cluster[0]["start_seconds"]),
                2,
            ),
            "confidence": max(
                float(event.get("confidence", 0.0)) for event in cluster
            ),
            "member_prediction_count": len(cluster),
        }
        for cluster in clusters
    ]


def evaluate_subject(
    subject_dir: Path,
    *,
    gap_seconds: float,
    policy: TemporalMatchingPolicy,
) -> dict[str, Any]:
    metrics = json.loads((subject_dir / "metrics.json").read_text(encoding="utf-8"))
    predictions = _generic_predictions(
        _read_jsonl(subject_dir / "event_predictions.jsonl"), gap_seconds
    )
    ground_truth = _generic_ground_truth(metrics, gap_seconds)
    matched, unmatched_predictions, unmatched_gt = match_temporal_events(
        ground_truth, predictions, policy
    )
    unique_gt = {
        (
            float(row["ground_truth"]["start_seconds"]),
            float(row["ground_truth"]["end_seconds"]),
        )
        for row in matched
    }
    tp = len(unique_gt)
    duplicate_predictions = max(0, len(matched) - tp)
    fp = len(unmatched_predictions) + duplicate_predictions
    fn = len(unmatched_gt)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / len(ground_truth) if ground_truth else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "subject_id": subject_dir.name,
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gt_events": len(ground_truth),
        "evaluated_minutes": metrics["evaluated_minutes"],
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(
            fp / float(metrics["evaluated_minutes"]), 4
        ),
        "matched_events": matched,
        "unmatched_predictions": unmatched_predictions,
        "unmatched_gt": unmatched_gt,
        "duplicate_predictions": duplicate_predictions,
    }


def evaluate_run(
    run_dir: Path,
    *,
    gap_seconds: float = 3.0,
    policy: TemporalMatchingPolicy = TemporalMatchingPolicy(),
) -> dict[str, Any]:
    aggregate = json.loads((run_dir / "event_matches.json").read_text(encoding="utf-8"))
    subject_results = {
        subject_id: evaluate_subject(
            run_dir / "subjects" / subject_id,
            gap_seconds=gap_seconds,
            policy=policy,
        )
        for subject_id in aggregate["cohort"]
    }
    tp = sum(row["true_positives"] for row in subject_results.values())
    fp = sum(row["false_positives"] for row in subject_results.values())
    fn = sum(row["false_negatives"] for row in subject_results.values())
    gt = sum(row["total_gt_events"] for row in subject_results.values())
    minutes = sum(float(row["evaluated_minutes"]) for row in subject_results.values())
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "schema_version": "1.0.0",
        "evaluation_type": "SECONDARY_GENERIC_PHONE_EPISODE_DIAGNOSTIC",
        "primary_v2_metric_replacement": False,
        "claim_status": "DIAGNOSTIC_ONLY_NOT_A_NEW_FROZEN_BENCHMARK",
        "source_run": str(run_dir),
        "gap_seconds": gap_seconds,
        "matching_policy": {
            "iou_threshold": policy.iou_threshold,
            "onset_tolerance_seconds": policy.onset_tolerance_seconds,
            "offset_tolerance_seconds": policy.offset_tolerance_seconds,
        },
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gt_events": gt,
        "evaluated_minutes": round(minutes, 2),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fp / minutes, 4) if minutes else 0.0,
        "per_subject": subject_results,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path)
    parser.add_argument("--gap-seconds", type=float, default=3.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = evaluate_run(args.run, gap_seconds=args.gap_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
