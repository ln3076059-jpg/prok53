"""Validate append-only live object decision logs without changing event metrics."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from statistics import mean, median
from typing import Any

from training.common import sha256_file


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def validate_decisions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    errors: list[str] = []
    active: dict[tuple[Any, ...], dict[str, Any]] = {}
    activation_latencies: list[float] = []
    decision_durations: list[float] = []
    session_finalizations = 0
    previous_time: float | None = None
    expected_sequence = 1

    for row in rows:
        sequence = int(row["sequence"])
        if sequence != expected_sequence:
            errors.append(
                f"sequence mismatch: expected {expected_sequence}, got {sequence}"
            )
        expected_sequence += 1
        decision_time = float(row["decision_time_seconds"])
        evidence_time = float(row["evidence_available_through_seconds"])
        if previous_time is not None and decision_time < previous_time:
            errors.append("decision times are not monotonic")
        previous_time = decision_time
        if evidence_time != decision_time:
            errors.append(f"evidence/decision time mismatch at sequence {sequence}")
        start = float(row["event_start_seconds"])
        if start > decision_time:
            errors.append(f"event starts after its decision at sequence {sequence}")
        key = (
            row.get("vehicle_context_id"),
            row.get("track_id"),
            row.get("occupant_group"),
        )
        decision_type = row["decision_type"]
        if decision_type == "START":
            if key in active:
                errors.append(f"duplicate START for active key at sequence {sequence}")
            active[key] = row
            activation_latencies.append(decision_time - start)
        elif decision_type == "END":
            opened = active.pop(key, None)
            if opened is None:
                errors.append(f"END without START at sequence {sequence}")
            else:
                if float(opened["event_start_seconds"]) != start:
                    errors.append(f"START/END event start mismatch at sequence {sequence}")
                decision_durations.append(decision_time - start)
            if row.get("reason") in {"MAX_FRAMES", "END_OF_STREAM"}:
                session_finalizations += 1
        else:
            errors.append(f"unsupported decision type {decision_type!r}")

    if active:
        errors.append(f"{len(active)} active event(s) lack END decisions")
    return {
        "status": "PASS" if not errors else "FAIL",
        "decisions": len(rows),
        "starts": sum(row["decision_type"] == "START" for row in rows),
        "ends": sum(row["decision_type"] == "END" for row in rows),
        "session_finalizations": session_finalizations,
        "activation_latency_seconds": {
            "mean": round(mean(activation_latencies), 6)
            if activation_latencies
            else None,
            "median": round(median(activation_latencies), 6)
            if activation_latencies
            else None,
            "maximum": round(max(activation_latencies), 6)
            if activation_latencies
            else None,
        },
        "decision_duration_seconds": {
            "mean": round(mean(decision_durations), 6)
            if decision_durations
            else None,
            "median": round(median(decision_durations), 6)
            if decision_durations
            else None,
            "maximum": round(max(decision_durations), 6)
            if decision_durations
            else None,
        },
        "errors": errors,
    }


def run(run_dir: Path, output_path: Path) -> dict[str, Any]:
    aggregate_path = run_dir / "event_matches.json"
    aggregate = json.loads(aggregate_path.read_text(encoding="utf-8"))
    per_subject: dict[str, Any] = {}
    for subject in aggregate["cohort"]:
        path = run_dir / "subjects" / subject / "object_decisions.jsonl"
        audit = validate_decisions(_read_jsonl(path))
        audit["path"] = str(path)
        audit["sha256"] = sha256_file(path)
        per_subject[subject] = audit
    baseline_path = Path("reports/v28_local_raw_d5/imgsz640/event_matches.json")
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))
    metric_keys = (
        "true_positives",
        "false_positives",
        "false_negatives",
        "precision",
        "recall",
        "f1",
    )
    metrics = {key: aggregate[key] for key in metric_keys}
    baseline_metrics = {key: baseline[key] for key in metric_keys}
    metric_reproduction = metrics == baseline_metrics
    all_logs_pass = all(row["status"] == "PASS" for row in per_subject.values())
    result = {
        "schema_version": "1.0.0",
        "created_at": datetime.now(UTC).isoformat(),
        "status": "PASS" if metric_reproduction and all_logs_pass else "FAIL",
        "source_run": str(run_dir),
        "source_event_matches_sha256": sha256_file(aggregate_path),
        "baseline_event_matches_sha256": sha256_file(baseline_path),
        "event_metrics": metrics,
        "baseline_event_metrics": baseline_metrics,
        "exact_metric_reproduction": metric_reproduction,
        "per_subject": per_subject,
        "claim_scope": "OBSERVABILITY_ONLY_DOES_NOT_CHANGE_EVENT_DECISIONS",
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run, args.output), indent=2))


if __name__ == "__main__":
    main()
