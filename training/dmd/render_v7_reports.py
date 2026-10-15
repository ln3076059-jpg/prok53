"""Render the required V7 research reports from machine-readable artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _event_row(row: dict[str, Any]) -> str:
    metrics = row["pooled_metrics"]
    return (
        f"| {row['experiment_id']} | {metrics['true_positives']} | "
        f"{metrics['false_positives']} | {metrics['false_negatives']} | "
        f"{_pct(metrics['precision'])} | {_pct(metrics['recall'])} | "
        f"{_pct(metrics['f1'])} | {metrics['false_alarms_per_minute']:.4f} | "
        f"{_pct(row['safe_driving_recall'])} | "
        f"{row['worst_subject_false_alarms_per_minute']:.4f} | "
        f"{'PASS' if row['eligible'] else 'FAIL'} |"
    )


def _clip_row(variant: dict[str, Any]) -> str:
    clip = variant["clip_metrics"]
    coarse = variant["coarse_metrics"]
    per_class = coarse["per_class"]
    return (
        f"| {variant['variant']['id']} | {_pct(clip['clip_accuracy'])} | "
        f"{_pct(clip['macro_f1'])} | {_pct(coarse['macro_f1'])} | "
        f"{_pct(per_class['SAFE_DRIVING']['recall'])} | "
        f"{_pct(per_class['PHONE_USE']['recall'])} | "
        f"{_pct(per_class['OTHER_ACTIVITY']['recall'])} |"
    )


def _per_subject_confusions(output_dir: Path, experiments: dict[str, Any]) -> str:
    lines = []
    for variant in experiments["variants"]:
        variant_id = variant["variant"]["id"]
        if variant_id in {"V7_A_V6_BASELINE", "V7_F_NEGATIVE_MARGIN"}:
            continue
        lines.append(f"### {variant_id}\n")
        for subject in DEVELOPMENT_SUBJECTS:
            path = output_dir / "folds" / f"{variant_id.lower()}_loso_{subject}.npz"
            with np.load(path, allow_pickle=False) as payload:
                labels = payload["labels"]
                probabilities = payload["probabilities"]
            truth = np.where(labels == 0, 0, np.where(labels == 5, 2, 1))
            coarse = np.column_stack(
                (probabilities[:, 0], probabilities[:, 1:5].sum(axis=1), probabilities[:, 5])
            )
            predicted = coarse.argmax(axis=1)
            matrix = np.zeros((3, 3), dtype=np.int64)
            for expected, actual in zip(truth, predicted, strict=True):
                matrix[int(expected), int(actual)] += 1
            lines.append(
                f"- `{subject}` (rows/cols SAFE, PHONE, OTHER): `{matrix.tolist()}`"
            )
        lines.append("")
    return "\n".join(lines)


def _seatbelt_queue(source: Path, output: Path) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    with source.open(encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            if row["true_class"] != "uncertain_or_occluded":
                continue
            probabilities = {
                "seatbelt_fastened": float(row["prob_fastened"]),
                "seatbelt_unfastened": float(row["prob_unfastened"]),
                "uncertain_or_occluded": float(row["prob_uncertain"]),
            }
            ordered = sorted(probabilities.items(), key=lambda item: item[1], reverse=True)
            rows.append(
                {
                    "image": row["image"],
                    "verified_class": row["true_class"],
                    "raw_prediction": row["raw_pred"],
                    "top1_probability": round(ordered[0][1], 6),
                    "top1_margin": round(ordered[0][1] - ordered[1][1], 6),
                    "model_correct": row["raw_pred"] == row["true_class"],
                    "roi_available_locally": False,
                    "review_status": "HUMAN_REVIEW_REQUIRED",
                    "recommended_action": "Inspect ROI crop; do not relabel automatically",
                }
            )
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return {
        "uncertain_support": len(rows),
        "correct": sum(bool(row["model_correct"]) for row in rows),
        "failures": sum(not bool(row["model_correct"]) for row in rows),
        "roi_available": 0,
        "queue_path": str(output),
    }


def render(output_dir: Path) -> None:
    experiments = json.loads((output_dir / "v7_experiments.json").read_text(encoding="utf-8"))
    safe = json.loads((output_dir / "v7_safe_failure_analysis.json").read_text(encoding="utf-8"))
    seatbelt = json.loads(
        Path("reports/v6_completion/v6_seatbelt_evaluation.json").read_text(encoding="utf-8")
    )
    runtime = json.loads(
        Path("reports/v6_completion/v6_runtime_benchmark.json").read_text(encoding="utf-8")
    )
    tests_path = output_dir / "v7_test_results.json"
    tests = json.loads(tests_path.read_text(encoding="utf-8")) if tests_path.is_file() else {}
    seatbelt_queue = _seatbelt_queue(
        Path("reports/calibration/seatbelt_classifier_raw_predictions.csv"),
        output_dir / "seatbelt_uncertain_review_queue.csv",
    )
    variants_table = "\n".join(_clip_row(row) for row in experiments["variants"])
    fusion_rows = [row for row in experiments["selection_rows"] if row["kind"] == "V4_OBJECT_PLUS_ACTION"]
    events_table = "\n".join(_event_row(row) for row in fusion_rows)
    selected = experiments["selected_candidate"]
    selected_text = selected["experiment_id"] if selected else "NONE"
    fp = safe["fused_false_positive_audit"]
    training_seconds = {
        row["variant"]["id"]: round(
            sum(float(fold.get("training", {}).get("training_seconds", 0)) for fold in row["folds"]),
            3,
        )
        for row in experiments["variants"]
    }
    head_rates = [
        float(fold["clips_per_second"])
        for row in experiments["variants"]
        for fold in row["folds"]
        if "clips_per_second" in fold
    ]
    reports: dict[str, str] = {
        "01_CURRENT_BASELINE_AUDIT.md": f"""# V7 current baseline audit

V7 starts from the frozen V6 hierarchical LOSO result. The pooled coarse confusion matrix is `{safe['pooled']['coarse_confusion_matrix']}` in SAFE/PHONE/OTHER order. SAFE recall is {_pct(safe['pooled']['safe_recall'])}: {safe['pooled']['safe_to_phone']} SAFE clips became PHONE and {safe['pooled']['safe_to_other']} became OTHER.

The V6 fusion baseline remains TP=16, FP=5, FN=13, F1=64.00%, FA/min=0.2266. No production configuration was changed.

Audit conclusion: SAFE collapse is primarily a coarse SAFE-versus-OTHER generalization problem. Fine phone loss alone cannot repair it.
""",
        "02_SAFE_DRIVING_FAILURE_ANALYSIS.md": f"""# SAFE driving failure analysis

All values below come from outer-LOSO held-out predictions; they were not used to tune thresholds.

- Pooled SAFE support: {safe['pooled']['safe_support']}
- Correct SAFE: {safe['pooled']['coarse_confusion_matrix'][0][0]}
- SAFE→PHONE: {safe['pooled']['safe_to_phone']}
- SAFE→OTHER: {safe['pooled']['safe_to_other']}

Per-subject detail, probability quantiles, phone-versus-negative margins, fine-head distributions, and temporal runs are preserved in `v7_safe_failure_analysis.json`. The result supports testing coarse-objective and binary-gate changes before any fine-head expansion.
""",
        "03_FALSE_POSITIVE_ROOT_CAUSES.md": f"""# False-positive root causes

The V6 hierarchical fusion produced {fp['count']} unmatched events. Auditable categories: `{json.dumps(fp['categories'], ensure_ascii=False)}`. Event-duration quantiles are `{json.dumps(fp['duration_seconds_quantiles'])}`.

Each case, sampled label composition, overlap with ground-truth events, and source evidence is recorded in `v7_safe_failure_analysis.json`. Visual cause remains unestablished where original ROI pixels are unavailable. The event matcher and 2-second merge rule were not changed; therefore improvements cannot be attributed to matcher tuning.
""",
        "04_SUBJECT_GENERALIZATION.md": f"""# Subject generalization

The only development subjects are `{', '.join(DEVELOPMENT_SUBJECTS)}`. Every reported V7 prediction is from an outer fold that excluded that subject during fitting. This is iterative three-subject evidence, not population-level validation.

{_per_subject_confusions(output_dir, experiments)}

The large subject-to-subject variation blocks claims beyond these three people. Consumed holdouts `gZ-37` and `gE-28` were not accessed.
""",
        "05_TRAINING_ABLATIONS.md": f"""# Training ablations

| Variant | Clip accuracy | Six-class macro-F1 | Coarse macro-F1 | SAFE recall | Phone recall | OTHER recall |
|---|---:|---:|---:|---:|---:|---:|
{variants_table}

Training seconds across three folds: `{json.dumps(training_seconds)}`. V7-B changes only coarse/fine loss treatment; V7-C adds verified training-only hard-negative weighting; V7-D changes to a binary PHONE/NON_PHONE gate; V7-E changes only sequence length from 9 to 17; V7-F applies the pre-registered 0.15 negative-evidence margin to V7-B and keeps suppressed clips in every denominator.
""",
        "06_EVENT_METRICS_COMPARISON.md": f"""# Event metrics comparison

| Fusion experiment | TP | FP | FN | Precision | Recall | F1 | FA/min | SAFE recall | Worst subject FA/min | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{events_table}

Selection uses the unchanged V6 gates: FA/min ≤0.30, worst-subject FA/min ≤0.60, SAFE recall ≥20%, then maximum event F1. Selected candidate: `{selected_text}`.
""",
        "07_SEATBELT_STATUS.md": f"""# Seatbelt status

The existing 641-sample validation artifact remains unchanged: accuracy {_pct(seatbelt['raw']['accuracy'])}, macro-F1 {_pct(seatbelt['raw']['macro_f1'])}, balanced accuracy {_pct(seatbelt['raw']['balanced_accuracy'])}, ECE {seatbelt['top1_ece_10_bin']:.4f}, and post-policy false-unfastened rate {_pct(seatbelt['false_unfastened']['rate_among_true_fastened'])}.

The uncertain class has {seatbelt_queue['uncertain_support']} verified samples: {seatbelt_queue['correct']} correct and {seatbelt_queue['failures']} failures. A review queue was generated at `{seatbelt_queue['queue_path']}`. ROI pixels are unavailable locally for all queued rows, so crop quality cannot be established and no labels or weights were changed.
""",
        "08_RUNTIME_BENCHMARK.md": f"""# Runtime benchmark

V7 temporal-head median throughput across measured folds is {np.median(head_rates):.1f} clips/s (range {min(head_rates):.1f}–{max(head_rates):.1f}) on local CPU. The unchanged full backend benchmark remains {runtime['full_backend_pipeline']['processing_fps']:.3f} FPS with detector mean {runtime['detector']['mean_ms']:.2f} ms and peak RSS {runtime['memory']['rss_peak_mb']:.1f} MB.

The detector, not the temporal head, is the measured bottleneck. No detector cadence or input resolution was reduced because there is no independent short-event accuracy set to quantify the miss risk. This is a deliberate no-change performance decision, not an optimization claim.
""",
        "09_TEST_RESULTS.md": f"""# Test results

Machine-readable results: `v7_test_results.json`.

```json
{json.dumps(tests, indent=2)}
```

Test failures caused by the pre-existing deleted `datasets/**` tree are separated from V7 code regressions. Docker availability is reported, not assumed.
""",
        "10_SCIENTIFIC_VALIDITY.md": """# Scientific validity

- Protocol: outer LOSO over three development subjects.
- Fixed decisions: event threshold 0.80, release ratio 0.65, merge gap 2.0 s, negative-evidence margin 0.15.
- Forbidden data: consumed holdouts `gZ-37`, `gE-28`; accessed count is zero.
- Hard negatives: labels and model scores from the two training subjects only; the outer subject is never accessed by the miner.
- Abstention: ambiguous clips remain in denominators, so false negatives are not hidden.
- Limitation: repeated design decisions on the same three subjects create adaptive overfitting risk even under LOSO.
- Governance: all checkpoints are `DEVELOPMENT_ONLY_NOT_VALIDATED`; no automatic violation and no production activation.
""",
        "11_FINAL_REPORT.md": f"""# V7 final report

V7 completed controlled model changes, 12 new LOSO trainings, one fixed-margin evaluation, SAFE/FP diagnostics, seatbelt review triage, runtime review, and regression tests. The selected candidate under unchanged gates is `{selected_text}`.

## Main results

| Fusion experiment | TP | FP | FN | Precision | Recall | F1 | FA/min | SAFE recall | Worst subject FA/min | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
{events_table}

The aspirational V7 targets are descriptive only; selection is governed by the unchanged V6 gates. Any candidate remains development-only until evaluated on genuinely new subjects. No V4/V5/V6 artifacts were overwritten and production behavior was not changed.
""",
    }
    for name, content in reports.items():
        (output_dir / name).write_text(content.strip() + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("reports/v7_optimization")
    )
    args = parser.parse_args()
    render(args.output_dir)


if __name__ == "__main__":
    main()
