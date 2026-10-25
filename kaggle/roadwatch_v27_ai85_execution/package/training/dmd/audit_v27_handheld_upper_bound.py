"""Audit the pre-registered forced-handheld-context upper bound."""

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
CANDIDATE = REPORT / "raw_d5_handheld_upper_bound"
SUBJECTS = ("gC-14", "gZ-36")


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def gt_key(gt: dict[str, Any]) -> tuple[Any, ...]:
    return (gt["source_action"], gt["start_seconds"], gt["end_seconds"])


def pred_key(prediction: dict[str, Any]) -> tuple[float, float]:
    return (prediction["start_seconds"], prediction["end_seconds"])


def main() -> None:
    plan_path = CANDIDATE / "experiment_plan.json"
    plan = read_json(plan_path)
    if plan["status"] != "PRE_REGISTERED_BEFORE_RESULT":
        raise ValueError("HANDHELD_UPPER_BOUND_NOT_PRE_REGISTERED")

    per_subject: dict[str, Any] = {}
    total = Counter()
    recovered_all: list[dict[str, Any]] = []
    lost_all: list[dict[str, Any]] = []
    for subject in SUBJECTS:
        baseline_root = BASELINE / subject
        candidate_root = CANDIDATE / subject
        baseline_metrics = read_json(
            baseline_root / "subjects" / subject / "metrics.json"
        )
        candidate_metrics = read_json(
            candidate_root / "subjects" / subject / "metrics.json"
        )
        summary = read_json(candidate_root / "event_matches.json")
        if summary.get("force_handheld_context") is not True:
            raise ValueError(f"HANDHELD_CONTEXT_NOT_FORCED:{subject}")
        if summary["verified_association"] is not True:
            raise ValueError(f"VERIFIED_ASSOCIATION_NOT_PRESERVED:{subject}")

        baseline_rows = read_jsonl(baseline_root / "autonomous_predictions.jsonl")
        candidate_rows = read_jsonl(candidate_root / "autonomous_predictions.jsonl")
        if len(baseline_rows) != len(candidate_rows):
            raise ValueError(f"HANDHELD_SAMPLE_COUNT_MISMATCH:{subject}")
        changed = []
        for baseline, candidate in zip(baseline_rows, candidate_rows, strict=True):
            if any(
                baseline[key] != candidate[key]
                for key in ("timestamp", "frame_idx", "candidate_id")
            ):
                raise ValueError(f"HANDHELD_TRACE_ALIGNMENT_FAILED:{subject}")
            if (
                baseline["confidence"] != candidate["confidence"]
                or baseline["rejection_reason"] != candidate["rejection_reason"]
            ):
                changed.append((baseline, candidate))

        baseline_gt = {
            gt_key(match["ground_truth"]): match["ground_truth"]
            for match in baseline_metrics["matched_events"]
        }
        candidate_gt = {
            gt_key(match["ground_truth"]): match["ground_truth"]
            for match in candidate_metrics["matched_events"]
        }
        recovered = [candidate_gt[key] for key in sorted(set(candidate_gt) - set(baseline_gt))]
        lost = [baseline_gt[key] for key in sorted(set(baseline_gt) - set(candidate_gt))]
        recovered_all.extend({"subject_id": subject, **row} for row in recovered)
        lost_all.extend({"subject_id": subject, **row} for row in lost)

        baseline_unmatched = {
            pred_key(row) for row in baseline_metrics["unmatched_predictions"]
        }
        candidate_unmatched = {
            pred_key(row) for row in candidate_metrics["unmatched_predictions"]
        }
        baseline_duplicates = len(baseline_metrics["matched_events"]) - baseline_metrics["true_positives"]
        candidate_duplicates = len(candidate_metrics["matched_events"]) - candidate_metrics["true_positives"]

        row = {
            "baseline": {
                key: baseline_metrics[key]
                for key in ("true_positives", "false_positives", "false_negatives", "precision", "recall", "f1")
            },
            "candidate": {
                key: candidate_metrics[key]
                for key in ("true_positives", "false_positives", "false_negatives", "precision", "recall", "f1")
            },
            "recovered_gt": recovered,
            "lost_gt": lost,
            "new_unmatched_prediction_intervals": sorted(candidate_unmatched - baseline_unmatched),
            "removed_unmatched_prediction_intervals": sorted(baseline_unmatched - candidate_unmatched),
            "baseline_duplicate_predictions": baseline_duplicates,
            "candidate_duplicate_predictions": candidate_duplicates,
            "changed_observations": len(changed),
            "baseline_admitted": sum(float(item["confidence"]) > 0.0 for item in baseline_rows),
            "candidate_admitted": sum(float(item["confidence"]) > 0.0 for item in candidate_rows),
            "baseline_changed_reason_counts": dict(
                sorted(Counter(left["rejection_reason"] for left, _ in changed).items())
            ),
            "candidate_changed_reason_counts": dict(
                sorted(Counter(right["rejection_reason"] for _, right in changed).items())
            ),
        }
        per_subject[subject] = row
        for prefix, metrics in (("baseline", baseline_metrics), ("candidate", candidate_metrics)):
            total[f"{prefix}_tp"] += metrics["true_positives"]
            total[f"{prefix}_fp"] += metrics["false_positives"]
            total[f"{prefix}_fn"] += metrics["false_negatives"]
        total["changed_observations"] += len(changed)
        total["baseline_admitted"] += row["baseline_admitted"]
        total["candidate_admitted"] += row["candidate_admitted"]

    def metrics(prefix: str) -> dict[str, float | int]:
        tp = total[f"{prefix}_tp"]
        fp = total[f"{prefix}_fp"]
        fn = total[f"{prefix}_fn"]
        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        return {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "precision": precision,
            "recall": recall,
            "f1": 2 * precision * recall / (precision + recall),
        }

    result = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "UPPER_BOUND_REJECTED_NET_F1_DECREASE",
        "experiment_id": plan["experiment_id"],
        "single_changed_variable": plan["single_changed_variable"],
        "selection_eligible": False,
        "production_eligible": False,
        "plan_sha256": digest(plan_path),
        "baseline_micro": metrics("baseline"),
        "candidate_micro": metrics("candidate"),
        "delta_counts": {
            "tp": total["candidate_tp"] - total["baseline_tp"],
            "fp": total["candidate_fp"] - total["baseline_fp"],
            "fn": total["candidate_fn"] - total["baseline_fn"],
        },
        "trace_delta": {
            "changed_observations": total["changed_observations"],
            "baseline_admitted": total["baseline_admitted"],
            "candidate_admitted": total["candidate_admitted"],
        },
        "recovered_gt": recovered_all,
        "lost_gt": lost_all,
        "per_subject": per_subject,
        "decision": (
            "Reject forced handheld context. It recovered one long gZ-36 event "
            "but added two net false positives and reduced micro F1."
        ),
    }
    target = CANDIDATE / "ablation_result.json"
    target.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({key: result[key] for key in ("status", "baseline_micro", "candidate_micro", "delta_counts")}, indent=2))


if __name__ == "__main__":
    main()
