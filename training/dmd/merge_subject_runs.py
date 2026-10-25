"""Merge completed per-subject V21 artifacts into one reproducible run.

Long CPU benchmarks persist every subject before writing their aggregate.  This
utility lets an interrupted multi-subject run be resumed without recomputing the
subjects that already completed.  It deliberately refuses duplicate subjects
and incompatible run configurations.
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any, Iterable


JSONL_FILES = (
    "autonomous_predictions.jsonl",
    "phone_candidate_records.jsonl",
    "action_predictions.jsonl",
    "event_predictions.jsonl",
)
CONFIG_KEYS = (
    "detector_mode",
    "detector_imgsz",
    "adaptive_sampling",
    "adaptive_stride",
    "adaptive_hold_seconds",
    "action_mode",
    "action_causal_mode",
    "missing_checkpoint_policy",
    "device",
    "force_handheld_context",
    "verified_association",
    "action_gating",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _configuration(aggregate: dict[str, Any]) -> dict[str, Any]:
    return {key: aggregate.get(key) for key in CONFIG_KEYS}


def _safe_copy_subject(source: Path, destination: Path) -> None:
    source_resolved = source.resolve()
    destination_resolved = destination.resolve()
    if source_resolved == destination_resolved:
        return
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite subject artifact: {destination}")
    shutil.copytree(source, destination)


def merge_subject_runs(
    source_runs: Iterable[Path],
    output_dir: Path,
) -> dict[str, Any]:
    sources = [Path(path) for path in source_runs]
    if not sources:
        raise ValueError("At least one source run is required")

    aggregates: list[dict[str, Any]] = []
    subjects: dict[str, tuple[Path, dict[str, Any]]] = {}
    for run_dir in sources:
        aggregate_path = run_dir / "event_matches.json"
        if aggregate_path.is_file():
            aggregates.append(_read_json(aggregate_path))
        subject_root = run_dir / "subjects"
        if not subject_root.is_dir():
            raise FileNotFoundError(f"Missing subject directory: {subject_root}")
        for subject_dir in sorted(path for path in subject_root.iterdir() if path.is_dir()):
            subject_id = subject_dir.name
            if subject_id in subjects:
                raise ValueError(f"Duplicate subject across source runs: {subject_id}")
            metrics_path = subject_dir / "metrics.json"
            if not metrics_path.is_file():
                raise FileNotFoundError(f"Missing subject metrics: {metrics_path}")
            metrics = _read_json(metrics_path)
            if int(metrics["true_positives"]) + int(metrics["false_negatives"]) != int(
                metrics["total_gt_events"]
            ):
                raise ValueError(f"Invalid GT accounting for {subject_id}")
            subjects[subject_id] = (subject_dir, metrics)

    if not aggregates:
        raise FileNotFoundError("No source run contains event_matches.json configuration")
    template = aggregates[0]
    expected_config = _configuration(template)
    for aggregate in aggregates[1:]:
        actual_config = _configuration(aggregate)
        if actual_config != expected_config:
            raise ValueError(
                "Incompatible run configurations: "
                f"expected {expected_config}, got {actual_config}"
            )

    output_dir.mkdir(parents=True, exist_ok=True)
    output_subject_root = output_dir / "subjects"
    output_subject_root.mkdir(parents=True, exist_ok=True)
    ordered_subjects = sorted(subjects)
    for subject_id in ordered_subjects:
        source_dir, _ = subjects[subject_id]
        _safe_copy_subject(source_dir, output_subject_root / subject_id)

    tp = sum(int(subjects[s][1]["true_positives"]) for s in ordered_subjects)
    fp = sum(int(subjects[s][1]["false_positives"]) for s in ordered_subjects)
    fn = sum(int(subjects[s][1]["false_negatives"]) for s in ordered_subjects)
    gt = sum(int(subjects[s][1]["total_gt_events"]) for s in ordered_subjects)
    minutes = sum(float(subjects[s][1]["evaluated_minutes"]) for s in ordered_subjects)
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    merged = {
        **template,
        "cohort": ordered_subjects,
        "total_gt_events": gt,
        "evaluated_minutes": round(minutes, 2),
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fp / minutes, 4) if minutes else 0.0,
        "per_subject": {
            subject_id: {
                "tp": metrics["true_positives"],
                "fp": metrics["false_positives"],
                "fn": metrics["false_negatives"],
                "precision": metrics["precision"],
                "recall": metrics["recall"],
                "f1": metrics["f1"],
                "fa_per_min": metrics["false_alarms_per_minute"],
                "effective_fps": metrics["effective_fps"],
                "action_provenance": metrics.get("action_provenance"),
                "sampling_diagnostics": metrics.get("sampling_diagnostics"),
            }
            for subject_id, (_, metrics) in (
                (subject_id, subjects[subject_id]) for subject_id in ordered_subjects
            )
        },
        "merge_provenance": {
            "source_runs": [str(path) for path in sources],
            "subject_artifacts_reused": True,
        },
    }
    (output_dir / "event_matches.json").write_text(
        json.dumps(merged, indent=2) + "\n", encoding="utf-8"
    )

    for filename in JSONL_FILES:
        with (output_dir / filename).open("w", encoding="utf-8", newline="\n") as handle:
            for subject_id in ordered_subjects:
                subject_file = output_subject_root / subject_id / filename
                if not subject_file.is_file():
                    raise FileNotFoundError(f"Missing subject artifact: {subject_file}")
                text = subject_file.read_text(encoding="utf-8")
                handle.write(text)
                if text and not text.endswith("\n"):
                    handle.write("\n")

    return merged


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    merged = merge_subject_runs(args.source_run, args.output)
    print(json.dumps(merged, indent=2))


if __name__ == "__main__":
    main()
