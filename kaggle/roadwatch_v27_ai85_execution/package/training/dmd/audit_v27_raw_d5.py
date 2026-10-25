"""Audit the completed V27 raw D5 detector/association/temporal ablation."""

from __future__ import annotations

import hashlib
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2


ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "v27_ai85_execution"
RUN = REPORT / "raw_d5_full_none"
SUBJECTS = {
    "gC-14": {
        "frames": 11916,
        "samples": 795,
        "video": ROOT / "dataset/external_dmd/subjects/gC-14/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_body.mp4",
    },
    "gZ-36": {
        "frames": 15351,
        "samples": 1024,
        "video": ROOT / "dataset/external_dmd/subjects/gZ-36/gZ_36_s2_2019-04-09T10;39;38+02;00_rgb_body.mp4",
    },
}
TRACE_FILES = (
    "autonomous_predictions.jsonl",
    "phone_candidate_records.jsonl",
    "action_predictions.jsonl",
    "event_predictions.jsonl",
)


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def digest(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def interval_rows(
    rows: list[dict[str, Any]], start: float, end: float
) -> list[dict[str, Any]]:
    return [row for row in rows if start <= float(row["timestamp"]) <= end]


def diagnose_fn(
    subject: str,
    index: int,
    gt: dict[str, Any],
    observations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    start = float(gt["start_seconds"])
    end = float(gt["end_seconds"])
    sampled = interval_rows(observations, start, end)
    proposals = interval_rows(candidates, start, end)
    selected = [row for row in sampled if row.get("candidate_id") is not None]
    admitted = [row for row in sampled if float(row.get("confidence", 0.0)) > 0.0]
    rejection_counts = Counter(row["rejection_reason"] for row in sampled)
    association_counts = Counter(row["assoc_state"] for row in proposals)
    association_reason_counts = Counter(
        reason for row in proposals for reason in row.get("reasons", [])
    )

    if not proposals:
        root_cause = "NO_PHONE_PROPOSAL_IN_GT_INTERVAL"
    elif (
        rejection_counts.get("UNVERIFIED_OCCUPANT_ASSOCIATION", 0)
        >= max(1, (len(selected) + 1) // 2)
        and len(admitted) < 2
    ):
        root_cause = "ASSOCIATION_UNKNOWN_REJECTED_MOST_OR_ALL_EVIDENCE"
    elif not admitted and len(proposals) < max(2, len(sampled) // 4):
        root_cause = "SPARSE_PROPOSAL_AND_ALL_SELECTED_EVIDENCE_REJECTED"
    elif not admitted and association_counts.get("UNKNOWN", 0):
        root_cause = "ASSOCIATION_UNKNOWN_REJECTED_ALL_EVIDENCE"
    elif not admitted:
        root_cause = "CONTEXT_OR_ASSOCIATION_REJECTED_ALL_EVIDENCE"
    elif len(admitted) < 2:
        root_cause = "INSUFFICIENT_ADMITTED_SUPPORT_FOR_TEMPORAL_ACTIVATION"
    else:
        root_cause = "TEMPORAL_FORMATION_OR_MATCH_FAILURE"

    return {
        "fn_id": f"{subject}-fn-{index:02d}",
        "subject_id": subject,
        "source_action": gt["source_action"],
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": float(gt["duration_seconds"]),
        "shorter_than_one_second": float(gt["duration_seconds"]) < 1.0,
        "sampled_observations": len(sampled),
        "proposal_records": len(proposals),
        "selected_candidate_observations": len(selected),
        "admitted_observations": len(admitted),
        "rejection_counts": dict(sorted(rejection_counts.items())),
        "association_counts": dict(sorted(association_counts.items())),
        "association_reason_counts": dict(sorted(association_reason_counts.items())),
        "root_cause": root_cause,
        "ground_truth": gt,
    }


def diagnose_fp(
    subject: str,
    index: int,
    prediction: dict[str, Any],
    observations: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> dict[str, Any]:
    start = float(prediction["start_seconds"])
    end = float(prediction["end_seconds"])
    sampled = interval_rows(observations, start, end)
    proposals = interval_rows(candidates, start, end)
    admitted = [row for row in sampled if float(row.get("confidence", 0.0)) > 0.0]
    return {
        "fp_id": f"{subject}-fp-{index:02d}",
        "subject_id": subject,
        "start_seconds": start,
        "end_seconds": end,
        "duration_seconds": float(prediction["duration"]),
        "sampled_observations": len(sampled),
        "proposal_records": len(proposals),
        "admitted_observations": len(admitted),
        "proposal_source_counts": dict(
            sorted(Counter(row["source"] for row in proposals).items())
        ),
        "association_counts": dict(
            sorted(Counter(row["assoc_state"] for row in proposals).items())
        ),
        "association_reason_counts": dict(
            sorted(
                Counter(
                    reason for row in proposals for reason in row.get("reasons", [])
                ).items()
            )
        ),
        "maximum_proposal_confidence": max(
            (float(row["confidence"]) for row in proposals), default=0.0
        ),
        "prediction": prediction,
    }


def main() -> None:
    subject_metrics: dict[str, dict[str, Any]] = {}
    fn_rows: list[dict[str, Any]] = []
    fp_rows: list[dict[str, Any]] = []
    artifacts: list[dict[str, Any]] = []
    coverage_by_subject: dict[str, dict[str, Any]] = {}

    for subject, expected in SUBJECTS.items():
        subject_root = RUN / subject
        metrics_path = subject_root / "subjects" / subject / "metrics.json"
        metrics = read_json(metrics_path)
        observations = read_jsonl(subject_root / "autonomous_predictions.jsonl")
        candidates = read_jsonl(subject_root / "phone_candidate_records.jsonl")
        actions = read_jsonl(subject_root / "action_predictions.jsonl")

        cap = cv2.VideoCapture(str(expected["video"]))
        source_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        source_fps = float(cap.get(cv2.CAP_PROP_FPS))
        cap.release()
        if source_frames != expected["frames"]:
            raise ValueError(f"SOURCE_FRAME_COUNT_MISMATCH:{subject}")
        if metrics["processed_frames"] != expected["samples"]:
            raise ValueError(f"INCOMPLETE_SAMPLED_DECODE:{subject}")
        if len(observations) != expected["samples"]:
            raise ValueError(f"OBSERVATION_COUNT_MISMATCH:{subject}")
        if actions:
            raise ValueError(f"ACTION_ABLATION_NOT_NONE:{subject}")

        run_summary = read_json(subject_root / "event_matches.json")
        required = {
            "detector_mode": "D5",
            "action_mode": "NONE",
            "device": "cpu",
            "verified_association": True,
            "action_gating": False,
        }
        for key, value in required.items():
            if run_summary.get(key) != value:
                raise ValueError(f"RAW_D5_CONFIG_MISMATCH:{subject}:{key}")

        for index, gt in enumerate(metrics["unmatched_gt"], 1):
            fn_rows.append(
                diagnose_fn(subject, index, gt, observations, candidates)
            )
        for index, prediction in enumerate(metrics["unmatched_predictions"], 1):
            fp_rows.append(
                diagnose_fp(subject, index, prediction, observations, candidates)
            )

        subject_metrics[subject] = {
            "source_frames": source_frames,
            "source_fps": source_fps,
            "source_seconds": source_frames / source_fps,
            "processed_frames": metrics["processed_frames"],
            "wall_time_seconds": metrics["wall_time_seconds"],
            "effective_sampled_fps": metrics["effective_fps"],
            "predicted_events": metrics["predicted_events_count"],
            "tp": metrics["true_positives"],
            "fp": metrics["false_positives"],
            "fn": metrics["false_negatives"],
            "gt": metrics["total_gt_events"],
            "precision": metrics["precision"],
            "recall": metrics["recall"],
            "f1": metrics["f1"],
            "fa_per_min": metrics["false_alarms_per_minute"],
        }
        coverage_by_subject[subject] = {
            "sampled_observations": len(observations),
            "observations_with_selected_candidate": sum(
                row.get("candidate_id") is not None for row in observations
            ),
            "admitted_observations": sum(
                float(row.get("confidence", 0.0)) > 0.0 for row in observations
            ),
            "rejection_counts": dict(
                sorted(Counter(row["rejection_reason"] for row in observations).items())
            ),
            "proposal_records": len(candidates),
            "proposal_source_counts": dict(
                sorted(Counter(row["source"] for row in candidates).items())
            ),
            "association_counts": dict(
                sorted(Counter(row["assoc_state"] for row in candidates).items())
            ),
            "association_reason_counts": dict(
                sorted(
                    Counter(
                        reason
                        for row in candidates
                        for reason in row.get("reasons", [])
                    ).items()
                )
            ),
        }
        for relative in ("event_matches.json", *TRACE_FILES):
            path = subject_root / relative
            artifacts.append(
                {
                    "path": str(path.relative_to(ROOT)),
                    "bytes": path.stat().st_size,
                    "sha256": digest(path),
                }
            )
        artifacts.append(
            {
                "path": str(metrics_path.relative_to(ROOT)),
                "bytes": metrics_path.stat().st_size,
                "sha256": digest(metrics_path),
            }
        )

    tp = sum(row["tp"] for row in subject_metrics.values())
    fp = sum(row["fp"] for row in subject_metrics.values())
    fn = sum(row["fn"] for row in subject_metrics.values())
    gt = sum(row["gt"] for row in subject_metrics.values())
    precision = tp / (tp + fp)
    recall = tp / gt
    f1 = 2 * precision * recall / (precision + recall)
    minutes = sum(row["source_seconds"] for row in subject_metrics.values()) / 60.0
    total_samples = sum(row["processed_frames"] for row in subject_metrics.values())
    total_wall = sum(row["wall_time_seconds"] for row in subject_metrics.values())

    aggregate = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "status": "RAW_D5_FULL_DEVELOPMENT_COMPLETE",
        "selection_eligible": False,
        "independent_test_claim": False,
        "cohort": list(SUBJECTS),
        "config": {
            "detector_mode": "D5",
            "action_mode": "NONE",
            "action_gating": False,
            "verified_association": True,
            "sample_stride": 15,
            "device": "cpu",
        },
        "per_subject": subject_metrics,
        "micro": {
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "gt": gt,
            "evaluated_minutes": round(minutes, 4),
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "false_alarms_per_minute": fp / minutes,
        },
        "gate": {
            "threshold": "precision > 0.85 AND recall > 0.85 AND f1 > 0.85",
            "status": "DEVELOPMENT_TARGET_NOT_MET",
            "minimum_feasible_counts_on_20_gt": {"tp": 18, "max_fp": 3, "max_fn": 2},
            "current_gap": {"additional_tp_needed": 4, "additional_fp_allowed": 0},
        },
        "comparison_warning": (
            "Cached E2 and raw D5 differ in execution path; their delta is not a "
            "single-variable causal estimate."
        ),
    }
    fn_waterfall = {
        "status": "RAW_TRACE_DIAGNOSTIC",
        "total_false_negatives": len(fn_rows),
        "shorter_than_one_second": sum(row["shorter_than_one_second"] for row in fn_rows),
        "root_cause_counts": dict(sorted(Counter(row["root_cause"] for row in fn_rows).items())),
        "events": fn_rows,
    }
    fp_ledger = {
        "status": "RAW_TRACE_DIAGNOSTIC",
        "total_false_positives": len(fp_rows),
        "events": fp_rows,
    }
    runtime = {
        "status": "FULL_TWO_SUBJECT_RAW_RUNTIME",
        "total_source_seconds": sum(row["source_seconds"] for row in subject_metrics.values()),
        "total_wall_time_seconds": total_wall,
        "source_video_realtime_throughput": (
            sum(row["source_seconds"] for row in subject_metrics.values()) / total_wall
        ),
        "total_sampled_frames": total_samples,
        "effective_sampled_fps": total_samples / total_wall,
        "per_subject": subject_metrics,
        "scope_limit": "CPU, stride-15 raw video; no action model and no per-stage p95 instrumentation",
    }
    coverage = {
        "status": "RAW_D5_MODE_COVERAGE",
        "action_mode": "NONE",
        "action_forward_count": 0,
        "per_subject": coverage_by_subject,
        "aggregate": {
            "sampled_observations": sum(
                row["sampled_observations"] for row in coverage_by_subject.values()
            ),
            "observations_with_selected_candidate": sum(
                row["observations_with_selected_candidate"]
                for row in coverage_by_subject.values()
            ),
            "admitted_observations": sum(
                row["admitted_observations"] for row in coverage_by_subject.values()
            ),
            "proposal_records": sum(
                row["proposal_records"] for row in coverage_by_subject.values()
            ),
        },
    }
    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "artifacts": artifacts,
    }

    outputs = {
        "aggregate_metrics.json": aggregate,
        "fn_waterfall.json": fn_waterfall,
        "fp_ledger.json": fp_ledger,
        "runtime.json": runtime,
        "mode_coverage.json": coverage,
        "artifact_manifest.json": manifest,
    }
    for name, value in outputs.items():
        (RUN / name).write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"micro": aggregate["micro"], "fn_roots": fn_waterfall["root_cause_counts"]}, indent=2))


if __name__ == "__main__":
    main()
