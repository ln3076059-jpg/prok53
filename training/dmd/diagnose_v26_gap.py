"""Recompute local forensic evidence; no inference, fitting or label changes."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

from training.dmd.action_checkpoint import sha256_file
from training.dmd.audit_v26_optimization import (
    OUTPUT,
    ROOT,
    build_fn_ledger,
    build_fp_ledger,
    full_raw_video_failure_ledger,
    full_raw_video_metrics,
    now,
    read_json,
    write_json,
)


def main() -> None:
    inputs: set[Path] = set()

    def read(path: Path):
        inputs.add(path)
        return read_json(path)

    def rows(path: Path):
        inputs.add(path)
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]

    detector_sources = {}
    training = {}
    full_dir = OUTPUT / "experiments/V26-FULL-D2"
    for subject, raw_run in (("gC-14", "gc14_run"), ("gZ-36", "gz36_run_final")):
        cache = ROOT / "reports/v22_verified_ai85/private" / subject
        summary = read(cache / "inference_summary.json")
        cached = rows(cache / "trace.jsonl")
        raw = rows(full_dir / raw_run / "phone_candidate_records.jsonl")
        detector_sources[subject] = {
            "cached_inference_summary": summary,
            "cached_candidate_sources": dict(
                Counter(candidate["source"] for row in cached for candidate in row["candidates"])
            ),
            "raw_mode": "D2",
            "raw_candidate_sources": dict(Counter(row["source"] for row in raw)),
            "interpretation": (
                "Different detectors and post-export filters; not bbox recall or parity."
            ),
        }
        manifest = read(
            ROOT
            / "models/candidate/v6_action_ablation_001"
            / f"hierarchical_loss_only_loso_{subject}.manifest.json"
        )
        fold = read(
            ROOT / "reports/v6_completion/folds" / f"hierarchical_loss_only_loso_{subject}.json"
        )
        training[subject] = {
            "training_subject_ids": manifest["training_subject_ids"],
            "validation_subject_ids": manifest["validation_subject_ids"],
            "preprocessing": manifest["feature_preprocessing"],
            "samples": fold["training"]["samples"],
            "last_training_epoch": fold["training"]["history"][-1],
            "interpretation": (
                "Training accuracy is not heldout accuracy; overfit remains a hypothesis."
            ),
        }

    corrected = full_dir / "gz36_run_seatfix2"
    candidates = rows(corrected / "phone_candidate_records.jsonl")
    predictions = rows(corrected / "autonomous_predictions.jsonl")
    metric = read(corrected / "subjects/gZ-36/metrics.json")
    matches = read(corrected / "event_matches.json")
    fn_ledger, waterfall = build_fn_ledger()
    fp_ledger = build_fp_ledger()
    for run in ("E2", "E3", "E4", "E5"):
        for name in ("metrics.json", "raw_predictions.jsonl", "event_matches.json"):
            inputs.add(OUTPUT / "experiments" / run / name)
    for run, subject in (("gc14_run", "gC-14"), ("gz36_run_final", "gZ-36")):
        inputs.add(full_dir / run / "autonomous_predictions.jsonl")
        inputs.add(full_dir / run / "subjects" / subject / "metrics.json")
    inputs.update(
        {
            ROOT / "models/temporal_config_v2.json",
            ROOT / "training/dmd/temporal_eval.py",
            ROOT / "training/dmd/audit_v26_optimization.py",
            ROOT / "training/dmd/run_autonomous_v21.py",
            Path(__file__),
        }
    )
    evidence = {
        "generated_at": now(),
        "scope": "LOCAL_DEVELOPMENT_FORENSICS_NOT_INDEPENDENT_HOLDOUT",
        "fn_ledger": fn_ledger,
        "fn_waterfall": waterfall,
        "fp_summary": {k: v for k, v in fp_ledger.items() if k != "entries"},
        "detector_source_mismatch": detector_sources,
        "raw_before_seat_fix": full_raw_video_failure_ledger(),
        "raw_after_seat_fix": {
            "status": full_raw_video_metrics()["status"],
            "subject": "gZ-36",
            "metrics": {
                k: metric[k]
                for k in (
                    "true_positives",
                    "false_positives",
                    "false_negatives",
                    "precision",
                    "recall",
                    "f1",
                )
            },
            "association_states": dict(Counter(row["assoc_state"] for row in candidates)),
            "frame_rejection_reasons": dict(
                Counter(row["rejection_reason"] for row in predictions)
            ),
            "matches": matches,
            "code_status": (
                "This run predates current boundary collector; no collector gain measured."
            ),
        },
        "training_evidence": training,
        "strict_target": {
            "ground_truth_count": 20,
            "threshold": 0.85,
            "acceptable_counts": [
                {
                    "tp": tp,
                    "max_fp": max(fp for fp in range(21) if tp / (tp + fp) > 0.85),
                    "fn": 20 - tp,
                }
                for tp in range(1, 21)
                if tp / 20 > 0.85
            ],
            "warning": (
                "Counts constrain this development cohort only; they do not prove generalization."
            ),
        },
        "input_sha256": {str(p.relative_to(ROOT)): sha256_file(p) for p in sorted(inputs)},
    }
    target = OUTPUT / "diagnosis/evidence.json"
    write_json(target, evidence)
    print(
        json.dumps(
            {
                "output": str(target.relative_to(ROOT)),
                "fn": waterfall["unmatched_by_primary_root_cause"],
                "fp": fp_ledger["false_positive_reasons"],
                "raw_corrected": evidence["raw_after_seat_fix"]["metrics"],
                "target": evidence["strict_target"]["acceptable_counts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
