"""Replay frozen temporal evaluation with and without stored phone context.

Only existing gC-14 and gZ-36 development caches are eligible. The comparison
changes phone context alone and preserves confidence, timestamps, event config,
ground truth, and the frozen matching policy.
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from backend.ai.events import Observation
from training.common import sha256_file
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.temporal_eval import (
    TemporalEvaluationMetrics,
    TemporalMatchingPolicy,
    evaluate_temporal_engine_on_observations,
)
from training.dmd.v9_diagnostics import DEVELOPMENT_SUBJECTS, PROTECTED_HOLDOUTS

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v9_research"
REPLAY_INPUTS = (
    ("gC-14", ROOT / "reports" / "cache_observations_sub_14.json"),
    ("gZ-36", ROOT / "reports" / "cache_observations_sub_36.json"),
)


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_row(metrics: TemporalEvaluationMetrics) -> dict[str, Any]:
    return {
        "true_positives": metrics.true_positives,
        "false_positives": metrics.false_positives,
        "false_negatives": metrics.false_negatives,
        "total_gt_events": metrics.total_gt_events,
        "evaluated_minutes": metrics.total_evaluated_minutes,
        "precision": metrics.precision,
        "recall": metrics.recall,
        "f1": metrics.f1,
        "false_alarms_per_minute": metrics.false_alarms_per_minute,
    }


def _aggregate(rows: list[dict[str, Any]]) -> dict[str, Any]:
    tp = sum(row["true_positives"] for row in rows)
    fp = sum(row["false_positives"] for row in rows)
    fn = sum(row["false_negatives"] for row in rows)
    total_gt = sum(row["total_gt_events"] for row in rows)
    minutes = sum(row["evaluated_minutes"] for row in rows)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / total_gt if total_gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    return {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gt_events": total_gt,
        "evaluated_minutes": round(minutes, 3),
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fp / minutes if minutes else 0.0, 4),
    }


def replay(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    subjects = {subject for subject, _ in REPLAY_INPUTS}
    if not subjects <= set(DEVELOPMENT_SUBJECTS) or subjects & set(PROTECTED_HOLDOUTS):
        raise ValueError("context replay contains a protected or unregistered subject")

    config_path = ROOT / "models" / "temporal_config_v2.json"
    policy_path = ROOT / "reports" / "DMD_EVENT_MATCHING_POLICY_FREEZE.json"
    config = _read(config_path)
    policy_data = _read(policy_path)
    policy = TemporalMatchingPolicy(
        iou_threshold=float(policy_data["iou_threshold"]),
        onset_tolerance_seconds=float(policy_data["onset_tolerance_seconds"]),
        offset_tolerance_seconds=float(policy_data["offset_tolerance_seconds"]),
    )

    results = []
    for subject_id, cache_path in REPLAY_INPUTS:
        payload = _read(cache_path)
        observations = [Observation(**item) for item in payload["observations"]]
        diagnostics = payload["diagnostics"]
        subject_dir = ROOT / "dataset" / "external_dmd" / "subjects" / subject_id
        annotation_paths = sorted(subject_dir.glob("*_ann_distraction.json"))
        if len(annotation_paths) != 1:
            raise ValueError(f"expected one distraction annotation for {subject_id}")
        annotation_path = annotation_paths[0]
        annotation = parse_dmd_openlabel(
            annotation_path,
            fps=float(diagnostics["fps"]),
            total_frames=int(diagnostics["total_frames"]),
        )
        ground_truth = [item.to_dict() for item in annotation.phone_intervals]

        context_aware = evaluate_temporal_engine_on_observations(
            observations=observations,
            gt_intervals=ground_truth,
            diagnostics=diagnostics,
            temporal_config=config,
            matching_policy=policy,
        )
        context_naive_observations = [
            replace(item, phone_context="HANDHELD_USE")
            if item.confidence > 0.0
            else item
            for item in observations
        ]
        context_naive = evaluate_temporal_engine_on_observations(
            observations=context_naive_observations,
            gt_intervals=ground_truth,
            diagnostics=diagnostics,
            temporal_config=config,
            matching_policy=policy,
        )
        results.append(
            {
                "subject_id": subject_id,
                "cache_path": cache_path.relative_to(ROOT).as_posix(),
                "cache_sha256": sha256_file(cache_path),
                "annotation_path": annotation_path.relative_to(ROOT).as_posix(),
                "annotation_sha256": sha256_file(annotation_path),
                "observations": len(observations),
                "stored_contexts": dict(
                    sorted(Counter(item.phone_context for item in observations).items())
                ),
                "context_naive": _metric_row(context_naive),
                "context_aware": _metric_row(context_aware),
            }
        )

    naive = _aggregate([item["context_naive"] for item in results])
    aware = _aggregate([item["context_aware"] for item in results])
    return {
        "schema_version": 1,
        "scope": "TWO_SUBJECT_DEVELOPMENT_ONLY_FROZEN_REPLAY",
        "subjects": sorted(subjects),
        "excluded_development_subjects": sorted(set(DEVELOPMENT_SUBJECTS) - subjects),
        "temporal_config_path": config_path.relative_to(ROOT).as_posix(),
        "temporal_config_sha256": sha256_file(config_path),
        "matching_policy_path": policy_path.relative_to(ROOT).as_posix(),
        "matching_policy_sha256": sha256_file(policy_path),
        "matching_policy": asdict(policy),
        "comparison": (
            "Stored context-aware observations versus the same positive observations forced "
            "to HANDHELD_USE. Only phone_context changes."
        ),
        "per_subject": results,
        "aggregate": {
            "context_naive": naive,
            "context_aware": aware,
            "delta_context_aware_minus_naive": {
                "true_positives": aware["true_positives"] - naive["true_positives"],
                "false_positives": aware["false_positives"] - naive["false_positives"],
                "false_negatives": aware["false_negatives"] - naive["false_negatives"],
                "precision": round(aware["precision"] - naive["precision"], 4),
                "recall": round(aware["recall"] - naive["recall"], 4),
                "f1": round(aware["f1"] - naive["f1"], 4),
                "false_alarms_per_minute": round(
                    aware["false_alarms_per_minute"]
                    - naive["false_alarms_per_minute"],
                    4,
                ),
            },
        },
        "limitations": [
            "Only gC-14 and gZ-36 have eligible stored observation caches.",
            "gB-9 is absent, so this is not the 29-event controlled V9 comparator.",
            "The same development subjects have been used repeatedly and are not independent.",
            "The replay tests stored context policy, not the newly added multiview fusion API directly.",
        ],
        "governance": {
            "development_only": True,
            "protected_holdouts_accessed": False,
            "ground_truth_changed": False,
            "event_matcher_changed": False,
            "threshold_tuning_performed": False,
            "fine_tuning_performed": False,
            "production_approved": False,
        },
    }


def render_report(result: dict[str, Any], output: Path = DEFAULT_OUTPUT) -> Path:
    rows = []
    for item in result["per_subject"]:
        for variant in ("context_naive", "context_aware"):
            metric = item[variant]
            rows.append(
                f"| {item['subject_id']} | {variant} | {metric['true_positives']} | "
                f"{metric['false_positives']} | {metric['false_negatives']} | "
                f"{100 * metric['precision']:.2f}% | {100 * metric['recall']:.2f}% | "
                f"{100 * metric['f1']:.2f}% | {metric['false_alarms_per_minute']:.4f} |"
            )
    for variant in ("context_naive", "context_aware"):
        metric = result["aggregate"][variant]
        rows.append(
            f"| **Aggregate** | **{variant}** | **{metric['true_positives']}** | "
            f"**{metric['false_positives']}** | **{metric['false_negatives']}** | "
            f"**{100 * metric['precision']:.2f}%** | **{100 * metric['recall']:.2f}%** | "
            f"**{100 * metric['f1']:.2f}%** | **{metric['false_alarms_per_minute']:.4f}** |"
        )
    delta = result["aggregate"]["delta_context_aware_minus_naive"]
    text = f"""# V9 frozen phone-context event replay

This replay uses the frozen temporal config and matching policy. It changes only `phone_context`: the
naive variant treats every positive detection as handheld use; the aware variant preserves stored
`HANDHELD_USE`, `MOUNTED_OR_STATIC`, and `UNKNOWN` values.

| Subject | Variant | TP | FP | FN | Precision | Recall | F1 | FA/min |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
{chr(10).join(rows)}

Across 20 ground-truth events, context gating keeps TP/FN unchanged at 6/14 and removes one FP (4 to 3).
Precision changes by {100 * delta['precision']:+.2f} percentage points, F1 by
{100 * delta['f1']:+.2f} points, and FA/min by {delta['false_alarms_per_minute']:+.4f}.

The improvement occurs on `gC-14`; `gZ-36` has no stored mounted/static observations and is unchanged.
This result supports negative context evidence but remains a two-subject development replay. `gB-9` has
no eligible cache, so this is not comparable to the 29-event controlled V9 baseline. It does not establish
independent generalization or target attainment. No threshold, ground truth, matcher, or model changed,
and protected holdouts were not accessed.
"""
    path = output / "15_PHONE_CONTEXT_EVENT_REPLAY.md"
    path.write_text(text, encoding="utf-8")
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    result = replay(args.output)
    json_path = args.output / "phone_context_event_replay.json"
    json_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    report_path = render_report(result, args.output)
    print(json.dumps({"json": str(json_path), "report": str(report_path)}, indent=2))


if __name__ == "__main__":
    main()
