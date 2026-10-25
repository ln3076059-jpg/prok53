"""Compare detector-source runs and audit evidence flow for each ground-truth event."""

from __future__ import annotations

import argparse
import json
import statistics
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


JsonObject = dict[str, Any]


def _read_json(path: Path) -> JsonObject:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[JsonObject]:
    if not path.is_file():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _rounded(value: float) -> float:
    return round(float(value), 6)


def _numeric_summary(values: Iterable[float]) -> JsonObject:
    ordered = sorted(float(value) for value in values)
    if not ordered:
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "mean": None,
            "p90": None,
            "maximum": None,
        }
    p90_index = round(0.9 * (len(ordered) - 1))
    return {
        "count": len(ordered),
        "minimum": _rounded(ordered[0]),
        "median": _rounded(statistics.median(ordered)),
        "mean": _rounded(statistics.fmean(ordered)),
        "p90": _rounded(ordered[p90_index]),
        "maximum": _rounded(ordered[-1]),
    }


def _bbox_summary(candidates: Iterable[JsonObject]) -> JsonObject:
    widths: list[float] = []
    heights: list[float] = []
    areas: list[float] = []
    for candidate in candidates:
        box = candidate.get("box")
        if not isinstance(box, list) or len(box) != 4:
            continue
        width = max(float(box[2]) - float(box[0]), 0.0)
        height = max(float(box[3]) - float(box[1]), 0.0)
        widths.append(width)
        heights.append(height)
        areas.append(width * height)
    return {
        "width": _numeric_summary(widths),
        "height": _numeric_summary(heights),
        "area": _numeric_summary(areas),
    }


def _duration_bucket(duration: float) -> str:
    if duration < 1.0:
        return "lt_1s"
    if duration < 3.0:
        return "1_to_3s"
    if duration < 10.0:
        return "3_to_10s"
    return "gte_10s"


def _in_interval(row: JsonObject, start: float, end: float) -> bool:
    timestamp = float(row.get("timestamp", -1.0))
    return start <= timestamp <= end


def _ground_truth_key(ground_truth: JsonObject) -> tuple[float, float, str]:
    return (
        float(ground_truth["start_seconds"]),
        float(ground_truth["end_seconds"]),
        str(ground_truth.get("source_action", "")),
    )


def _deduplicated_matches(metrics: JsonObject) -> tuple[list[JsonObject], list[JsonObject]]:
    grouped: dict[tuple[float, float, str], list[JsonObject]] = {}
    for match in metrics.get("matched_events", []):
        grouped.setdefault(_ground_truth_key(match["ground_truth"]), []).append(match)
    best_matches: list[JsonObject] = []
    duplicate_matches: list[JsonObject] = []
    for matches in grouped.values():
        ordered = sorted(matches, key=lambda row: float(row["iou"]), reverse=True)
        best_matches.append(ordered[0])
        duplicate_matches.extend(ordered[1:])
    return best_matches, duplicate_matches


def _event_rows(metrics: JsonObject) -> list[tuple[JsonObject, JsonObject | None]]:
    best_matches, _ = _deduplicated_matches(metrics)
    rows: list[tuple[JsonObject, JsonObject | None]] = [
        (match["ground_truth"], match) for match in best_matches
    ]
    rows.extend((ground_truth, None) for ground_truth in metrics.get("unmatched_gt", []))
    return sorted(rows, key=lambda item: float(item[0]["start_seconds"]))


def _analyze_event(
    ground_truth: JsonObject,
    match: JsonObject | None,
    observations: list[JsonObject],
    candidates: list[JsonObject],
    candidate_by_id: dict[str, JsonObject],
) -> JsonObject:
    start = float(ground_truth["start_seconds"])
    end = float(ground_truth["end_seconds"])
    event_observations = [row for row in observations if _in_interval(row, start, end)]
    event_candidates = [row for row in candidates if _in_interval(row, start, end)]
    selected_ids = [
        str(row["candidate_id"])
        for row in event_observations
        if row.get("candidate_id") is not None
    ]
    selected_sources = Counter(
        candidate_by_id[candidate_id].get("source", "UNKNOWN")
        for candidate_id in selected_ids
        if candidate_id in candidate_by_id
    )
    admitted = [row for row in event_observations if float(row.get("confidence", 0.0)) > 0.0]
    duration = end - start
    result: JsonObject = {
        "ground_truth": ground_truth,
        "duration_seconds": _rounded(duration),
        "duration_bucket": _duration_bucket(duration),
        "matched": match is not None,
        "sample_count": len(event_observations),
        "proposal_count": len(event_candidates),
        "selected_count": len(selected_ids),
        "admitted_count": len(admitted),
        "proposal_sources": dict(sorted(Counter(
            row.get("source", "UNKNOWN") for row in event_candidates
        ).items())),
        "selected_sources": dict(sorted(selected_sources.items())),
        "rejection_reasons": dict(sorted(Counter(
            row.get("rejection_reason", "UNKNOWN") for row in event_observations
        ).items())),
        "association_states": dict(sorted(Counter(
            row.get("assoc_state", "UNKNOWN") for row in event_candidates
        ).items())),
        "proposal_confidence": _numeric_summary(
            float(row.get("confidence", 0.0)) for row in event_candidates
        ),
        "proposal_bbox": _bbox_summary(event_candidates),
    }
    if match is not None:
        prediction = match["prediction"]
        prediction_duration = (
            float(prediction["end_seconds"]) - float(prediction["start_seconds"])
        )
        result["match"] = {
            "iou": _rounded(match["iou"]),
            "start_latency_seconds": _rounded(match["start_latency"]),
            "end_latency_seconds": _rounded(match["end_latency"]),
            "prediction_duration_seconds": _rounded(prediction_duration),
            "duration_ratio": _rounded(prediction_duration / max(duration, 1e-9)),
            "prediction": prediction,
        }
    return result


def _boundary_summary(metrics: JsonObject) -> JsonObject:
    matches, _ = _deduplicated_matches(metrics)
    duration_ratios: list[float] = []
    for match in matches:
        ground_truth = match["ground_truth"]
        prediction = match["prediction"]
        gt_duration = float(ground_truth["end_seconds"]) - float(
            ground_truth["start_seconds"]
        )
        pred_duration = float(prediction["end_seconds"]) - float(
            prediction["start_seconds"]
        )
        duration_ratios.append(pred_duration / max(gt_duration, 1e-9))
    return {
        "temporal_iou": _numeric_summary(float(row["iou"]) for row in matches),
        "absolute_start_error_seconds": _numeric_summary(
            abs(float(row["start_latency"])) for row in matches
        ),
        "absolute_end_error_seconds": _numeric_summary(
            abs(float(row["end_latency"])) for row in matches
        ),
        "prediction_to_gt_duration_ratio": _numeric_summary(duration_ratios),
    }


def _event_boundary_summary(events: list[JsonObject]) -> JsonObject:
    matches = [event["match"] for event in events if event.get("matched")]
    return {
        "temporal_iou": _numeric_summary(float(row["iou"]) for row in matches),
        "absolute_start_error_seconds": _numeric_summary(
            abs(float(row["start_latency_seconds"])) for row in matches
        ),
        "absolute_end_error_seconds": _numeric_summary(
            abs(float(row["end_latency_seconds"])) for row in matches
        ),
        "prediction_to_gt_duration_ratio": _numeric_summary(
            float(row["duration_ratio"]) for row in matches
        ),
    }


def _duration_recall(events: list[JsonObject]) -> JsonObject:
    buckets: dict[str, dict[str, int]] = {
        name: {"ground_truth": 0, "matched": 0}
        for name in ("lt_1s", "1_to_3s", "3_to_10s", "gte_10s")
    }
    for event in events:
        bucket = buckets[event["duration_bucket"]]
        bucket["ground_truth"] += 1
        bucket["matched"] += int(event["matched"])
    for bucket in buckets.values():
        support = bucket["ground_truth"]
        bucket["recall"] = _rounded(bucket["matched"] / support) if support else None
    return buckets


def analyze_subject(subject_dir: Path) -> JsonObject:
    metrics = _read_json(subject_dir / "metrics.json")
    observations = _read_jsonl(subject_dir / "autonomous_predictions.jsonl")
    candidates = _read_jsonl(subject_dir / "phone_candidate_records.jsonl")
    candidate_by_id = {
        str(row["candidate_id"]): row
        for row in candidates
        if row.get("candidate_id") is not None
    }
    events = [
        _analyze_event(ground_truth, match, observations, candidates, candidate_by_id)
        for ground_truth, match in _event_rows(metrics)
    ]
    _, duplicate_matches = _deduplicated_matches(metrics)
    false_positive_ledger = [
        {"reason": "UNMATCHED_PREDICTION", "prediction": prediction}
        for prediction in metrics.get("unmatched_predictions", [])
    ] + [
        {
            "reason": "DUPLICATE_GT_MATCH",
            "prediction": match["prediction"],
            "ground_truth": match["ground_truth"],
            "iou": match["iou"],
        }
        for match in duplicate_matches
    ]
    if len(events) != int(metrics["total_gt_events"]):
        raise ValueError(
            f"GT ledger mismatch in {subject_dir}: {len(events)} != "
            f"{metrics['total_gt_events']}"
        )
    if len(false_positive_ledger) != int(metrics["false_positives"]):
        raise ValueError(
            f"FP ledger mismatch in {subject_dir}: {len(false_positive_ledger)} != "
            f"{metrics['false_positives']}"
        )
    selected_ids = [
        str(row["candidate_id"])
        for row in observations
        if row.get("candidate_id") is not None
    ]
    return {
        "event_metrics": {
            key: metrics.get(key)
            for key in (
                "true_positives",
                "false_positives",
                "false_negatives",
                "total_gt_events",
                "precision",
                "recall",
                "f1",
                "false_alarms_per_minute",
                "effective_fps",
                "wall_time_seconds",
            )
        },
        "evidence_totals": {
            "samples": len(observations),
            "proposals": len(candidates),
            "selected": len(selected_ids),
            "admitted": sum(
                float(row.get("confidence", 0.0)) > 0.0 for row in observations
            ),
            "proposal_sources": dict(sorted(Counter(
                row.get("source", "UNKNOWN") for row in candidates
            ).items())),
            "selected_sources": dict(sorted(Counter(
                candidate_by_id[candidate_id].get("source", "UNKNOWN")
                for candidate_id in selected_ids
                if candidate_id in candidate_by_id
            ).items())),
            "rejection_reasons": dict(sorted(Counter(
                row.get("rejection_reason", "UNKNOWN") for row in observations
            ).items())),
            "association_states": dict(sorted(Counter(
                row.get("assoc_state", "UNKNOWN") for row in candidates
            ).items())),
            "proposal_confidence": _numeric_summary(
                float(row.get("confidence", 0.0)) for row in candidates
            ),
            "proposal_bbox": _bbox_summary(candidates),
        },
        "duration_recall": _duration_recall(events),
        "boundary_quality": _boundary_summary(metrics),
        "ground_truth_events": events,
        "false_positive_events": false_positive_ledger,
    }


def analyze_run(run_dir: Path) -> JsonObject:
    aggregate = _read_json(run_dir / "event_matches.json")
    subjects = {
        subject_id: analyze_subject(run_dir / "subjects" / subject_id)
        for subject_id in aggregate["cohort"]
    }
    all_events = [
        event
        for subject in subjects.values()
        for event in subject["ground_truth_events"]
    ]
    source_counts: Counter[str] = Counter()
    selected_source_counts: Counter[str] = Counter()
    rejection_counts: Counter[str] = Counter()
    association_counts: Counter[str] = Counter()
    for subject in subjects.values():
        totals = subject["evidence_totals"]
        source_counts.update(totals["proposal_sources"])
        selected_source_counts.update(totals["selected_sources"])
        rejection_counts.update(totals["rejection_reasons"])
        association_counts.update(totals["association_states"])
    return {
        "run_dir": str(run_dir),
        "configuration": {
            key: aggregate.get(key)
            for key in (
                "detector_mode",
                "detector_imgsz",
                "action_mode",
                "action_causal_mode",
                "device",
                "force_handheld_context",
                "verified_association",
                "action_gating",
                "adaptive_sampling",
                "adaptive_stride",
                "adaptive_hold_seconds",
            )
        },
        "event_metrics": {
            key: aggregate.get(key)
            for key in (
                "true_positives",
                "false_positives",
                "false_negatives",
                "total_gt_events",
                "precision",
                "recall",
                "f1",
                "false_alarms_per_minute",
                "evaluated_minutes",
            )
        },
        "evidence_totals": {
            "samples": sum(row["evidence_totals"]["samples"] for row in subjects.values()),
            "proposals": sum(row["evidence_totals"]["proposals"] for row in subjects.values()),
            "selected": sum(row["evidence_totals"]["selected"] for row in subjects.values()),
            "admitted": sum(row["evidence_totals"]["admitted"] for row in subjects.values()),
            "proposal_sources": dict(sorted(source_counts.items())),
            "selected_sources": dict(sorted(selected_source_counts.items())),
            "rejection_reasons": dict(sorted(rejection_counts.items())),
            "association_states": dict(sorted(association_counts.items())),
        },
        "duration_recall": _duration_recall(all_events),
        "boundary_quality": _event_boundary_summary(all_events),
        "subjects": subjects,
    }


def compare_runs(run_specs: list[tuple[str, Path]]) -> JsonObject:
    if not run_specs:
        raise ValueError("At least one run must be provided")
    runs = {name: analyze_run(path) for name, path in run_specs}
    baseline_name = run_specs[0][0]
    baseline = runs[baseline_name]["event_metrics"]
    comparison = []
    for name, _ in run_specs:
        metrics = runs[name]["event_metrics"]
        comparison.append({
            "run": name,
            **metrics,
            "delta_tp": metrics["true_positives"] - baseline["true_positives"],
            "delta_fp": metrics["false_positives"] - baseline["false_positives"],
            "delta_fn": metrics["false_negatives"] - baseline["false_negatives"],
            "delta_precision": _rounded(metrics["precision"] - baseline["precision"]),
            "delta_recall": _rounded(metrics["recall"] - baseline["recall"]),
            "delta_f1": _rounded(metrics["f1"] - baseline["f1"]),
        })
    return {
        "schema_version": "1.0.0",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "baseline_run": baseline_name,
        "comparison": comparison,
        "runs": runs,
    }


def _parse_run_spec(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("Run must be NAME=PATH")
    name, raw_path = value.split("=", 1)
    if not name.strip() or not raw_path.strip():
        raise argparse.ArgumentTypeError("Run must be NAME=PATH")
    return name.strip(), Path(raw_path.strip())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run",
        action="append",
        required=True,
        type=_parse_run_spec,
        help="Run specification NAME=PATH; first run is the comparison baseline",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = compare_runs(args.run)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result["comparison"], indent=2))


if __name__ == "__main__":
    main()
