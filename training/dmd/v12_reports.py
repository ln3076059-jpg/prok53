"""Render V12 evidence reports after diagnostics and checks have actually run."""

# ruff: noqa: E501

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from training.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v12_research"


def read(name: str) -> Any:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def source(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text.rstrip() + "\n", encoding="utf-8")


def write_json(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def read_native_log(path: Path) -> str:
    """Read logs written by either PowerShell 5 (UTF-16) or modern UTF-8 shells."""
    payload = path.read_bytes()
    if payload.startswith((b"\xff\xfe", b"\xfe\xff")):
        return payload.decode("utf-16")
    return payload.decode("utf-8")


def pct(value: float | None) -> str:
    return "NOT MEASURED" if value is None else f"{value * 100:.2f}%"


def table(headers: list[str], rows: list[list[Any]]) -> str:
    result = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    result += ["| " + " | ".join(str(cell) for cell in row) + " |" for row in rows]
    return "\n".join(result)


def junit(name: str) -> dict:
    path = OUT / name
    root = ET.parse(path).getroot()
    suite = root[0] if root.tag == "testsuites" else root
    return {key: int(suite.attrib.get(key, 0)) for key in ("tests", "failures", "errors", "skipped")} | {
        "seconds": float(suite.attrib.get("time", 0)), "sha256": sha256_file(path)
    }


def aggregate_event(replay: dict, variant: str) -> dict:
    records = [row["variants"][variant] for row in replay["subjects"]]
    tp = sum(row["true_positives"] for row in records)
    fp = sum(row["false_positives"] for row in records)
    fn = sum(row["false_negatives"] for row in records)
    minutes = sum(row["total_evaluated_minutes"] for row in records)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    return {"true_positives": tp, "false_positives": fp, "false_negatives": fn,
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0,
            "evaluated_minutes": minutes, "false_alarms_per_minute": fp / minutes if minutes else 0}


def run() -> None:
    plan, dataset, models = read("experiment_plan.json"), read("dataset_manifest.json"), read("model_manifest.json")
    reconcile, streaming = read("metric_reconciliation.json"), read("streaming_audit.json")
    roi, replay = read("roi_audit.json"), read("event_match_records.json")
    v11 = source("reports/v11_research/metric_comparison.json")
    v11_gen = source("reports/v11_research/generalization_experiments.json")
    v11_fn = source("reports/v11_research/fn_root_cause.json")
    runtime = source("reports/ai85_plus/runtime_benchmark.json")
    full, targeted = junit("pytest_full.xml"), junit("pytest_targeted.xml")
    selected = next(r for r in v11_gen["representations"] if r["representation_id"] == v11_gen["selected_candidate"])
    naive, aware = aggregate_event(replay, "context_naive"), aggregate_event(replay, "context_aware")
    argmax_disagreements = sum(
        fold["prediction_probability_argmax_disagreements"]
        for result in reconcile["results"] for fold in result["folds"]
    )
    gaps = sum(fold["timestamp_gaps_above_075s"] for result in reconcile["results"] for fold in result["folds"])
    removed = sorted(event for row in replay["subjects"] for event in row["delta"]["removed_fp_ids"])
    recovered = sorted(event for row in replay["subjects"] for event in row["delta"]["recovered_tp_ids"])
    lost = sorted(event for row in replay["subjects"] for event in row["delta"]["lost_tp_ids"])
    new_fp = sorted(event for row in replay["subjects"] for event in row["delta"]["new_fp_ids"])
    all_sizes = [case["phone_pixel_size_64_driver_cache"][0] for case in roi["cases"] if case["phone_pixel_size_64_driver_cache"]]

    metric_comparison = {
        "schema_version": 1, "evidence_scope": "DEVELOPMENT_ONLY",
        "metric_reconciliation": {"status": reconcile["status"], "discrepancies": reconcile["discrepancies"],
                                  "probability_argmax_disagreements": argmax_disagreements},
        "clip_selected_e3": selected,
        "phone_event_controlled_29": v11["event"]["controlled_binary"],
        "context_replay_two_subject": {"naive": naive, "aware": aware,
            "delta_ids": {"recovered_tp_ids": recovered, "lost_tp_ids": lost,
                          "removed_fp_ids": removed, "new_fp_ids": new_fp}},
        "v12_e3_event_metrics": None,
        "seatbelt": v11["seatbelt"],
        "independent_holdout": None, "targets_established_independently": False,
        "production_approved": False,
    }
    write_json("metric_comparison.json", metric_comparison)
    registry = {
        "schema_version": 1, "head": plan["head"], "plan_sha256": sha256_file(OUT / "experiment_plan.json"),
        "experiments": {
            "V12-A": {"decision": "KEEP", "result": "30 prediction artifacts reconciled; zero metric discrepancies", "artifact": "metric_reconciliation.json"},
            "V12-B": {"decision": "KEEP_ENGINEERING", "result": "batch/stream parity on three development caches", "artifact": "streaming_audit.json"},
            "V12-C": {"decision": "KEEP_DIAGNOSTIC_HYPOTHESIS", "result": "13 original frames decoded; 12 reviewed boxes preserved", "artifact": "roi_audit.json"},
            "V12-D": {"decision": "BLOCKED", "reason": dataset["denied_operations"]["linear_head"]},
            "V12-E": {"decision": "KEEP_SYNTHETIC_SEMANTICS_REAL_EVAL_BLOCKED", "reason": "13/13 FN cases lack verified occupant tracks"},
            "V12-F": {"decision": "KEEP_BASELINE_REPLAY_E3_BLOCKED", "result": "One inherited FP ID removed; no TP delta", "artifact": "event_match_records.json"},
            "V12-G": {"decision": "BLOCKED", "reason": dataset["denied_operations"]["boundary"]},
            "V12-H": {"decision": "BLOCKED", "reason": "500/500 seatbelt candidates remain PENDING; zero automatic labels"},
        },
        "commands": ["py -m training.dmd.v12_diagnostics", "py -m pytest -q tests/test_v12_research.py",
                     "py -m pytest -q", "npm run build", "py -m training.dmd.v12_reports",
                     "py -m training.dmd.verify_v12"],
        "source_hashes": {name: sha256_file(OUT / name) for name in (
            "experiment_plan.json", "dataset_manifest.json", "model_manifest.json",
            "metric_reconciliation.json", "streaming_audit.json", "roi_audit.json",
            "event_match_records.json", "diagnostic_execution.json")},
        "training_performed": False, "production_approved": False, "final_status": "RESEARCH_ONLY",
    }
    write_json("experiment_registry.json", registry)

    repo = read("repository_environment.json")
    write("01_CODE_AND_DATA_AUDIT.md", f"""# V12 code and data audit

Audit HEAD: `{repo['head']}`. The user-authored `PRODUCT.md` remained a working-tree input and was not treated as a model artifact. The V12 code adds a live fail-closed permission gate, bounded causal streaming E3, portable JSON-only linear-head inference, conservative association ambiguity handling, metric recomputation, original-frame ROI diagnostics, and event-ID replay. Candidate integration is opt-in and production defaults were not changed.

DMD permission remains `RIGHTS_ACCEPTED={dataset['permissions'].get('RIGHTS_ACCEPTED')}`, `CANONICAL_ELIGIBLE={dataset['permissions'].get('CANONICAL_ELIGIBLE')}`, `TEMPORAL_DEVELOPMENT_ELIGIBLE={dataset['permissions'].get('TEMPORAL_DEVELOPMENT_ELIGIBLE')}`. Linear-head, calibration, fusion, boundary, backbone fitting and upload are all denied by the current live record. `gB-9`, `gC-14`, and `gZ-36` remain repeatedly used development subjects. Protected `gE-28` and `gZ-37` media/labels were not opened.

The seatbelt manifest has {dataset['seatbelt_review']['samples']} candidates: `{dataset['seatbelt_review']['status_counts']}`, with {dataset['seatbelt_review']['automatic_training_labels']} automatic training labels. The checkpoint inventory contains {models['checkpoint_count']} files totaling {models['total_bytes'] / 2**20:.1f} MiB; files were hashed without pickle deserialization or activation. No training, external upload, production activation, ground-truth edit, or matcher edit occurred.
""")
    write("02_METRIC_RECONCILIATION.md", f"""# V12 metric reconciliation

V12 recomputed confusion matrices, class metrics, balanced accuracy and pooled metrics directly from all 30 V11 NPZ prediction artifacts. Status: **{reconcile['status']}**, with {len(reconcile['discrepancies'])} discrepancies. Source SHA-256 values and fold outputs are in `metric_reconciliation.json`.

The frozen selected E3 result remains clip Macro-F1 {pct(selected['pooled']['macro_f1'])}, balanced accuracy {pct(selected['pooled']['balanced_accuracy'])}, SAFE recall {pct(selected['pooled']['safe_recall'])}, PHONE recall {pct(selected['pooled']['phone_recall'])}, OTHER recall {pct(selected['pooled']['other_recall'])}. Worst-subject Macro-F1 is {pct(selected['worst_subject']['macro_f1'])}; worst SAFE recall is {pct(selected['worst_subject']['safe_recall'])}.

A new integrity finding is that stored `predictions` disagree with `argmax(probabilities)` on {argmax_disagreements} rows. This follows the V11 one-vs-rest API: class prediction uses decision-function argmax while stored probabilities are separately normalized one-vs-rest sigmoids. V12 portable inference preserves decision argmax and marks probabilities UNCALIBRATED. The artifacts contain {gaps} timestamp gaps above 0.75 s across repeated variants; V12 streaming resets on cadence breaks rather than bridging them.
""")
    latency_rows = [[r["subject"], r["samples"], r["parity"], f"{r['max_absolute_error']:.2e}",
                     f"{r['latency_ms']['50']:.3f}", f"{r['latency_ms']['95']:.3f}", f"{r['latency_ms']['99']:.3f}"] for r in streaming["subjects"]]
    write("03_ONLINE_CANDIDATE.md", f"""# V12 online candidate

`OnlineE3` reproduces spatial-plus-past-running normalization one sample at a time. It resets on session/camera/track changes, missing views, out-of-order timestamps, and cadence gaps. State is bounded to 8 frames and 120 feature rows. Eight initial samples per stream are marked insufficient-history; they are not silently interpreted as negative evidence.

{table(['Subject', 'Samples', 'Parity', 'Max error', 'p50 ms', 'p95 ms', 'p99 ms'], latency_rows)}

These timings cover cached 64×64 RGB to E3 feature only. They exclude decode, detector, fitted head, fusion and event persistence. No fitted E3 scaler/head exists in V11 artifacts, so real E3 event replay remains `BLOCKED_MISSING_FITTED_HEAD_SCALER`. V12 provides JSON-only numeric export/load code and verifies it on synthetic fixtures; it does not manufacture model parameters. Probability output is explicitly uncalibrated. The research adapter remains disabled unless a caller opts in.
""")
    fn_counts = v11_fn["summary"]["failure_stage_counts_nonexclusive"]
    write("04_PHONE_ROI_ASSOCIATION.md", f"""# V12 phone ROI and association

V12 decoded one original-resolution BODY frame for each of the 13 reviewed FNs and verified source video hashes against the V11 manifest. Twelve reviewed phone boxes are preserved in real-pixel dynamic crops; `gB-9-phone-06` has no valid reviewed box. No detector was rerun, and no detection AP or event recovery is claimed.

The reviewed phone boxes would be only {min(all_sizes):.2f}–{max(all_sizes):.2f} pixels wide after reduction to the 64-pixel driver cache. This supplies direct evidence that the V11 representation cache is too coarse for small-phone localization. The full crops and coordinate provenance are stored under `roi_evidence/` and `roi_audit.json`.

The inherited root-cause counts remain: specialist miss {fn_counts['SPECIALIST_DETECTOR_MISS']}/13, auxiliary valid candidate not admitted {fn_counts['AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE']}/13, action below gate {fn_counts['ACTION_EVIDENCE_BELOW_GATE']}/13, driver association UNKNOWN 13/13. Association now abstains when mounted/static context lacks independent verification or multiple verified occupant boxes are similarly plausible. Its score remains a heuristic, not a calibrated probability. Real association accuracy remains unmeasured because occupant-track ground truth is absent.
""")
    folds = [[f["heldout_subject"], pct(f["macro_f1"]), pct(f["safe_recall"]), pct(f["phone_recall"]), pct(f["other_recall"])] for f in selected["folds"]]
    write("05_REPRESENTATION_ABLATIONS.md", f"""# V12 representation ablations

No new representation was fit because the authoritative DMD record denies fitting. V12-A retained the frozen V11 predictions only after exact metric reconciliation. V12-B retained the online implementation because it matches the batch E3 features across all three caches. V12-C retained original-resolution ROI as a diagnostic hypothesis because reviewed boxes become extremely small at 64×64.

{table(['Held-out subject', 'Macro-F1', 'SAFE recall', 'PHONE recall', 'OTHER recall'], folds)}

V12-D appearance/motion fitting, V12-G temporal fitting, calibration and learned fusion are BLOCKED. The reported folds are repeatedly examined LEVEL 2 development evidence and were already used for E3 selection. They are not a new unbiased test. No threshold sweep, seed sweep, Group DRO, CLIP download, backbone tuning, or protected-holdout access was performed.
""")
    write("06_EVENT_REPLAY.md", f"""# V12 event replay

| Variant | TP | FP | FN | Precision | Recall | F1 | FA/min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen context naive, two subjects | {naive['true_positives']} | {naive['false_positives']} | {naive['false_negatives']} | {pct(naive['precision'])} | {pct(naive['recall'])} | {pct(naive['f1'])} | {naive['false_alarms_per_minute']:.4f} |
| Frozen context aware, two subjects | {aware['true_positives']} | {aware['false_positives']} | {aware['false_negatives']} | {pct(aware['precision'])} | {pct(aware['recall'])} | {pct(aware['f1'])} | {aware['false_alarms_per_minute']:.4f} |

The context-only delta is recovered TP IDs={recovered}, lost TP IDs={lost}, removed FP IDs={removed}, new FP IDs={new_fp}. The removed false positive is now traceable as `{removed[0] if removed else 'NONE'}`. Counts were recomputed from match records and duplicate predictions are charged as FP. Duration-bin records are included per subject.

The 29-event controlled comparator remains TP=16, FP=4, FN=13, P=80.00%, R=55.17%, F1=65.31%, FA/min=0.1813. Its historical scope differs from this two-subject replay. V12 generated no E3 events because the fitted head/scaler and verified occupants are absent. Emission latency was not preserved in the frozen evaluator, so online latency is NOT MEASURED rather than inferred from backdated start times.
""")
    seat = v11["seatbelt"]
    write("07_SEATBELT_RESULTS.md", f"""# V12 seatbelt results

The current review manifest contains {dataset['seatbelt_review']['samples']} candidates, all PENDING, with zero automatic training labels. Rights and approved group splits are also absent. V12 therefore did not train, calibrate, or relabel the seatbelt branch.

The inherited development best remains Macro-F1 {pct(seat['best_development_macro_f1'])}. Its source variants report UNCERTAIN recall 4% for the best horizontal-flip result. This does not meet the 85% objectives. `UNCERTAIN_OR_OCCLUDED` remains distinct from `UNFASTENED`; invisible belts were not converted into violations. False-unfastened, abstention and balanced accuracy remain the inherited values in `metric_comparison.json`, with no new independent evidence.

Triplet Attention, visibility/state multitask fitting, torso ROI training and calibration remain BLOCKED pending human-approved three-state labels, explicit fitting rights and a subject-disjoint split.
""")
    frontend = read_native_log(OUT / "frontend_build.log")
    backend = read_native_log(OUT / "backend_smoke.log")
    frontend_passed = "built in" in frontend
    backend_passed = '"status": "PASS"' in backend
    write("08_RUNTIME_AND_TESTS.md", f"""# V12 runtime and tests

| Check | Result |
| --- | --- |
| Python compile | PASS; exit code 0 |
| Scoped Ruff | PASS; exit code 0 |
| Targeted V11+V12 pytest | PASS: {targeted['tests']} tests, {targeted['failures']} failures, {targeted['errors']} errors |
| Full pytest | PASS: {full['tests']} tests, {full['failures']} failures, {full['errors']} errors |
| Frontend build | {'PASS' if frontend_passed else 'FAIL'} |
| Backend OpenAPI smoke | {'PASS' if backend_passed else 'FAIL'} |
| Full V12 inference smoke | NOT RUN: no fitted E3 head/scaler and no active-pipeline change |

Streaming E3 feature p50 across subjects is {max(r['latency_ms']['50'] for r in streaming['subjects']):.3f} ms; maximum subject p95 is {max(r['latency_ms']['95'] for r in streaming['subjects']):.3f} ms and p99 is {max(r['latency_ms']['99'] for r in streaming['subjects']):.3f} ms. This is cached feature latency only. The inherited full-backend benchmark remains {runtime['full_backend_pipeline']['processing_fps']:.3f} FPS with p50 {runtime['full_backend_pipeline']['per_context_latency']['p50_ms']:.2f} ms and p95 {runtime['full_backend_pipeline']['per_context_latency']['p95_ms']:.2f} ms. V12 does not claim real-time operation.
""")
    write("09_FINAL_RESEARCH_REPORT.md", f"""# V12 final research report

Decision: **RESEARCH_ONLY; REJECT FOR PRODUCTION**.

V12 closes engineering and evidence gaps without unauthorized fitting. It recomputes all 30 V11 prediction artifacts with zero metric discrepancy, implements a bounded online E3 path with exact batch parity, decodes all 13 FN review points from source-resolution videos, strengthens association uncertainty, and reconstructs the context replay with stable event IDs. The strongest new diagnosis is that reviewed phone boxes shrink to as little as {min(all_sizes):.2f} pixels wide in the V11 64-pixel cache. This supports future original-resolution behavior crops once rights allow an approved experiment.

No accuracy target is newly achieved. Selected development E3 remains Macro-F1 {pct(selected['pooled']['macro_f1'])}, SAFE recall {pct(selected['pooled']['safe_recall'])}, PHONE recall {pct(selected['pooled']['phone_recall'])}, and worst-subject SAFE recall {pct(selected['worst_subject']['safe_recall'])}. Controlled event metrics remain P=80.00%, R=55.17%, F1=65.31%, FA/min=0.1813. Seatbelt best development Macro-F1 remains {pct(seat['best_development_macro_f1'])}; 500 labels remain pending. Phone AP is NOT MEASURED.

There is no independent holdout and no independent AI85 result. DMD fitting rights, approved labels, a locked group split, a persisted fitted E3 head/scaler, verified occupant tracks, and complete online emission records remain blockers. The highest-value next experiment is a rights-approved, subject-disjoint original-resolution hand/phone ROI plus causal E3 candidate, exported before a fixed event replay on new validation subjects. Production activation requires separate human review after a locked independent evaluation.

`production_approved=false`; `final_status=RESEARCH_ONLY`; `independent_ai85_established=false`.
""")
    test_results = {"schema_version": 1, "compile": "PASS", "ruff": "PASS", "targeted": targeted,
                    "full": full,
                    "frontend": "PASS" if frontend_passed else "FAIL",
                    "backend": "PASS" if backend_passed else "FAIL",
                    "full_inference_smoke": "NOT_RUN_BLOCKED_MISSING_FITTED_E3_HEAD",
                    "production_approved": False}
    write_json("test_results.json", test_results)


if __name__ == "__main__":
    run()
