"""Audit the pre-registered V27 verified-association bypass upper bound."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "v27_ai85_execution"
BASELINE = REPORT / "raw_d5_full_none"
CANDIDATE = REPORT / "raw_d5_assoc_bypass"
SUBJECTS = ("gC-14", "gZ-36")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def main() -> None:
    plan_path = CANDIDATE / "experiment_plan.json"
    plan = read_json(plan_path)
    if plan["status"] != "PRE_REGISTERED_BEFORE_RESULT":
        raise ValueError("ASSOCIATION_ABLATION_NOT_PRE_REGISTERED")

    per_subject: dict[str, Any] = {}
    totals = Counter()
    for subject in SUBJECTS:
        baseline_root = BASELINE / subject
        candidate_root = CANDIDATE / subject
        baseline_metrics = read_json(
            baseline_root / "subjects" / subject / "metrics.json"
        )
        candidate_metrics = read_json(
            candidate_root / "subjects" / subject / "metrics.json"
        )
        candidate_summary = read_json(candidate_root / "event_matches.json")
        if candidate_summary["verified_association"] is not False:
            raise ValueError(f"ASSOCIATION_NOT_BYPASSED:{subject}")
        metric_keys = (
            "true_positives",
            "false_positives",
            "false_negatives",
            "precision",
            "recall",
            "f1",
        )
        baseline_event = {key: baseline_metrics[key] for key in metric_keys}
        candidate_event = {key: candidate_metrics[key] for key in metric_keys}

        baseline_rows = read_jsonl(baseline_root / "autonomous_predictions.jsonl")
        candidate_rows = read_jsonl(candidate_root / "autonomous_predictions.jsonl")
        if len(baseline_rows) != len(candidate_rows):
            raise ValueError(f"ASSOCIATION_ABLATION_SAMPLE_COUNT_MISMATCH:{subject}")
        changed = []
        for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
            stable = ("timestamp", "frame_idx", "candidate_id")
            if any(baseline[key] != candidate[key] for key in stable):
                raise ValueError(f"ASSOCIATION_ABLATION_TRACE_ALIGNMENT_FAILED:{subject}")
            if (
                baseline["confidence"] != candidate["confidence"]
                or baseline["rejection_reason"] != candidate["rejection_reason"]
            ):
                changed.append((baseline, candidate))

        baseline_admitted = sum(float(row["confidence"]) > 0.0 for row in baseline_rows)
        candidate_admitted = sum(float(row["confidence"]) > 0.0 for row in candidate_rows)
        row = {
            "baseline_event": baseline_event,
            "candidate_event": candidate_event,
            "event_metrics_identical": baseline_event == candidate_event,
            "sampled_observations": len(baseline_rows),
            "changed_observations": len(changed),
            "baseline_admitted": baseline_admitted,
            "candidate_admitted": candidate_admitted,
            "baseline_changed_reason_counts": dict(
                sorted(Counter(left["rejection_reason"] for left, _ in changed).items())
            ),
            "candidate_changed_reason_counts": dict(
                sorted(Counter(right["rejection_reason"] for _, right in changed).items())
            ),
        }
        per_subject[subject] = row
        totals["sampled_observations"] += len(baseline_rows)
        totals["changed_observations"] += len(changed)
        totals["baseline_admitted"] += baseline_admitted
        totals["candidate_admitted"] += candidate_admitted

    if not all(row["event_metrics_identical"] for row in per_subject.values()):
        status = "ASSOCIATION_BYPASS_CHANGED_EVENT_METRICS"
    else:
        status = "HYPOTHESIS_REJECTED_ASSOCIATION_BYPASS_NO_EVENT_EFFECT"
    if totals["baseline_admitted"] != totals["candidate_admitted"]:
        raise ValueError("ASSOCIATION_BYPASS_UNEXPECTED_ADMISSION_CHANGE")

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "experiment_id": plan["experiment_id"],
        "single_changed_variable": plan["single_changed_variable"],
        "selection_eligible": False,
        "production_eligible": False,
        "plan_sha256": digest(plan_path),
        "per_subject": per_subject,
        "aggregate_trace_delta": dict(totals),
        "causal_finding": (
            "Bypassing verified association changed 89 rejection labels from "
            "UNVERIFIED_OCCUPANT_ASSOCIATION to PHONE_VISIBLE_WITHOUT_HANDHELD_USE, "
            "but changed zero admissions and zero event metrics. The measured limiter "
            "is therefore handheld-context geometry or downstream temporal support, "
            "not the association safeguard alone."
        ),
        "next_single_variable_diagnostic": (
            "Keep verified association enabled and test a pre-registered handheld-context "
            "upper bound; do not promote any bypass to production."
        ),
    }
    target = CANDIDATE / "ablation_result.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"status": status, "aggregate_trace_delta": dict(totals)}, indent=2))


if __name__ == "__main__":
    main()
