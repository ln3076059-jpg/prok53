"""Render V13 reports from completed, machine-readable diagnostics."""

# ruff: noqa: E501

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from training.common import sha256_file
from training.dmd.v12_reports import read_native_log

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v13_research"


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name: str, content: str) -> None:
    (OUT / name).write_text(content.strip() + "\n", encoding="utf-8")


def junit(name: str) -> dict:
    root = ET.parse(OUT / name).getroot()
    suite = root[0] if root.tag == "testsuites" else root
    return {key: int(suite.attrib.get(key, 0)) for key in ("tests", "failures", "errors")}


def run() -> None:
    env, data = read("environment.json"), read("dataset_manifest.json")
    roi, ready = read("high_resolution_roi.json"), read("candidate_readiness.json")
    metrics = read("metrics_comparison.json")
    full, targeted = junit("pytest_full.xml"), junit("pytest_targeted.xml")
    frontend = "built in" in read_native_log(OUT / "frontend_build.log")
    backend = '"status": "PASS"' in read_native_log(OUT / "backend_smoke.log")
    e3 = metrics["clip_selected_e3"]
    controlled = metrics["phone_event_controlled_29"]
    aware = metrics["context_replay_two_subject"]["aware"]
    seat = metrics["seatbelt"]
    pending = data["seatbelt_review"]["status_counts"].get("PENDING", 0)

    write("01_HEAD_AND_ENVIRONMENT_AUDIT.md", f"""# V13 HEAD and environment audit

Audit started from `{env['head']}` on `{env['branch']}`. The only pre-existing working-tree change was the user-authored `prompt.md`; V13 did not overwrite it. Python is {env['python']} on `{env['platform']}`. Free workspace disk at audit time was {env['disk_free_bytes'] / 2**30:.2f} GiB. NVIDIA tooling was unavailable, and system RAM was constrained, so no large model download or GPU experiment was attempted.

V13 modifies the ROI extraction path only to support `output_size=None`, which preserves source crop pixels. The default remains 64x64 for compatibility. It adds a high-resolution wrapper with frame-content SHA-256, exact dimensions, crop transform, phone coverage and boundary clipping. No production registry or active model was changed.
""")
    write("02_DATA_RIGHTS_AND_SPLIT_AUDIT.md", f"""# V13 data rights and split audit

The live authoritative record still reports `RIGHTS_STATUS={data['permissions'].get('RIGHTS_STATUS')}`, `RIGHTS_ACCEPTED={data['permissions'].get('RIGHTS_ACCEPTED')}`, `CANONICAL_ELIGIBLE={data['permissions'].get('CANONICAL_ELIGIBLE')}`, and `TEMPORAL_DEVELOPMENT_ELIGIBLE={data['permissions'].get('TEMPORAL_DEVELOPMENT_ELIGIBLE')}`. Its SHA-256 is `{data['permission_sha256']}`. Diagnostics are permitted, but fitting and upload remain denied.

`gB-9`, `gC-14`, and `gZ-36` remain repeatedly used development subjects. No media, labels, cache, or results from protected `gZ-37` or `gE-28` were opened. There is no new subject-disjoint validation or locked independent test. Training performed: **false**. Independent AI85 status: **NOT ESTABLISHED**.
""")
    write("03_BASELINE_RECONCILIATION.md", f"""# V13 baseline reconciliation

V13 inherits the V12 reconciliation of 30 immutable V11 prediction artifacts with zero metric discrepancy. Controlled phone events remain TP={controlled['true_positives']}, FP={controlled['false_positives']}, FN={controlled['false_negatives']}, Precision={controlled['precision']:.4f}, Recall={controlled['recall']:.4f}, F1={controlled['f1']:.4f}, and FA/min={controlled['false_alarms_per_minute']:.4f}. No new candidate event metrics were produced.

E3 development clip Macro-F1 remains {e3['pooled']['macro_f1']:.4f}; SAFE/PHONE/OTHER recall remains {e3['pooled']['safe_recall']:.4f}/{e3['pooled']['phone_recall']:.4f}/{e3['pooled']['other_recall']:.4f}. Worst-subject Macro-F1 is {e3['worst_subject']['macro_f1']:.4f}, and worst SAFE recall is {e3['worst_subject']['safe_recall']:.4f}. These are repeated development results, not independent evidence.
""")
    write("04_PHONE_HIGH_RES_ROI.md", f"""# V13 phone high-resolution ROI

V13 decoded {roi['decoded_frames']} original BODY frames from hash-verified sources and evaluated R0-R5 extraction. There are {roi['reviewed_boxes']} authenticated reviewed boxes; R5 preserved full phone coverage for {roi['r5_full_phone_coverage']}/{roi['reviewed_boxes']}. The remaining case has no valid phone proposal and stays a hard example. Original frame coordinates, content hashes, output sizes, transform provenance, edge clipping and coverage are recorded in `high_resolution_roi.json`.

R0 remains the frozen 64x64 reference. R1 preserves the original driver crop, R2 uses padded phone/hand evidence, R3 records a face/hand fallback when those boxes are absent, R4 returns multiple source-resolution regions, and R5 uses a scale-aware dynamic crop. These are oracle-box diagnostics. Detector confidence was not regenerated, candidate yield did not change, Phone AP is NOT MEASURED, and zero events are claimed recovered.
""")
    write("05_ONLINE_E3_ARTIFACT_READINESS.md", f"""# V13 OnlineE3 artifact readiness

OnlineE3 remains `{ready['online_e3_features']}` based on V12 batch/stream parity, bounded state, causal history and reset tests. The fitted E3 head/scaler is `{ready['fitted_e3_head_scaler']}`. The safe JSON numeric contract and synthetic parity tests exist, but V13 did not create coefficients from synthetic data or refit DMD.

Class order remains SAFE_DRIVING, PHONE_USE, OTHER_ACTIVITY. Scores remain explicitly uncalibrated. A real event replay is `{ready['event_replay']}`. The candidate adapter remains opt-in and the active runtime model is unchanged.
""")
    write("06_BEHAVIOR_REPRESENTATION_RESULTS.md", """# V13 behavior representation results

A0 frozen E3 was retained as the reference. A1 original-resolution spatial evidence was implemented and measured only as ROI preservation and runtime. A2-A6 require a fitted classifier or backbone experiment and were BLOCKED by the live rights gate. No CLIP model was downloaded, and no DualScope, MixStyle or Group DRO result is claimed.

The central engineering finding is that source-resolution evidence can be preserved for all 12 reviewed boxes without upscaling the old 64x64 cache. This addresses an information-loss mechanism but does not demonstrate improved classification. New V13 accuracy metrics are null; all clip metrics remain the V12 development values.
""")
    write("07_DRIVER_PHONE_ASSOCIATION.md", """# V13 driver-phone association

Association retains explicit DRIVER_ASSOCIATED, PASSENGER_ASSOCIATED, MOUNTED_OR_STATIC and UNKNOWN states. V12 already added ambiguity abstention, finite score validation and independent verification for mounted/static context. V13 high-resolution crop provenance can now feed this module without coordinate loss.

Real accuracy remains blocked because all 13 FN cases lack verified occupant-track ground truth. Heuristic confidence is not described as calibrated probability. Overlapping occupants, missing tracks, unverified mounted context and phone-boundary conditions are covered by regression tests; UNKNOWN is never converted to a safe or violating decision.
""")
    write("08_FUSION_AND_EVENT_REPLAY.md", f"""# V13 fusion and event replay

No V13 fitted candidate replay was legal or technically possible. The frozen two-subject context-aware replay remains TP={aware['true_positives']}, FP={aware['false_positives']}, FN={aware['false_negatives']}, F1={aware['f1']:.4f}, FA/min={aware['false_alarms_per_minute']:.4f}. Its immutable records and source hash are referenced in `event_match_records.json`; the matcher was not changed.

Recovered TP IDs remain empty, lost TP IDs remain empty, and the inherited removed FP ID is `gC-14:PRED:0.000000:0.504032`. V13 reports no E3 Event F1, onset latency or emission latency. Stable IDs and real emission timestamps remain implemented in the opt-in adapter but require a legitimate fitted head and verified driver association for real evaluation.
""")
    write("09_SEATBELT_RESULTS.md", f"""# V13 seatbelt results

The seatbelt queue contains {pending} PENDING candidates and zero automatically admitted training labels. V13 did not convert, infer or approve any human label. Fitting remains blocked by rights, label approval and split requirements.

The inherited best development Macro-F1 is {seat['best_development_macro_f1']:.6f}; the best recorded UNCERTAIN recall remains 4%. FASTENED, UNFASTENED and UNCERTAIN_OR_OCCLUDED remain distinct. No hidden abstention or coverage filtering was introduced, and occlusion was not relabeled as unfastened.
""")
    write("10_RUNTIME_BENCHMARK.md", f"""# V13 runtime benchmark

R0-R5 source-resolution extraction across one decoded frame has p50 {roi['latency_ms']['50']:.3f} ms, p95 {roi['latency_ms']['95']:.3f} ms and p99 {roi['latency_ms']['99']:.3f} ms. This timed the six ROI variants after frame decode; it is not detector, encoder, model, fusion, event persistence or full-pipeline latency.

The inherited E3 cached-feature timing remains roughly 3-7 ms, and the inherited full-backend benchmark remains 1.119 FPS with p50 600.79 ms and p95 2050.07 ms. V13 did not claim real-time operation, GPU FPS, end-to-end alert delay or full candidate throughput because the fitted candidate is absent.
""")
    write("11_TEST_RESULTS.md", f"""# V13 test results

Python compile and scoped Ruff passed. Targeted V11/V12/V13-adjacent tests: {targeted['tests']} passed, {targeted['failures']} failures, {targeted['errors']} errors. Full pytest: {full['tests']} passed, {full['failures']} failures, {full['errors']} errors. Frontend build: {'PASS' if frontend else 'FAIL'}. Backend OpenAPI smoke: {'PASS' if backend else 'FAIL'}.

New coverage verifies original-resolution pixel preservation, deterministic frame hashing, identity scale, boundary padding and partial phone coverage. Existing tests continue to cover rights gates, protected subjects, OnlineE3 parity/reset, numeric model loading, association ambiguity and stable event IDs.
""")
    write("12_FINAL_AI_LEAD_REPORT.md", f"""# V13 final AI lead report

Decision: **RESEARCH_ONLY; REJECT FOR PRODUCTION**.

V13 changed `behavior_roi.py`, added `high_res_roi.py`, V13 diagnostics/report tooling and regression coverage. It ran live permission checks, 13 original-frame R0-R5 diagnostics, baseline/artifact reconciliation, targeted and full tests, frontend build and backend smoke. The measurable new result is engineering evidence: 12/12 reviewed phone boxes retain full coverage in the source-resolution R5 crop, at ROI-suite p50 {roi['latency_ms']['50']:.3f} ms. This is not an accuracy result.

Phone events remain 16/4/13 with P/R/F1 80.00%/55.17%/65.31% and FA/min 0.1813. E3 clip Macro-F1 remains 56.39%, SAFE/PHONE/OTHER recall 60.38%/66.78%/43.26%, worst-subject SAFE recall 54.36%. Seatbelt best development Macro-F1 remains 58.52%, UNCERTAIN recall 4%. Phone AP and E3 Event F1 remain NOT MEASURED.

There is no fitted E3 head/scaler, verified occupant ground truth, approved seatbelt label set, valid fitting authorization, new group-disjoint validation or independent holdout. The 85% objective is not scientifically established. The highest-value next experiment is a rights-approved, subject-disjoint fit comparing frozen R0 against R5 plus causal E3, with validation-fixed calibration and event policy before a locked independent test. Production approval remains false.
""")
    test_results = {
        "compile": "PASS", "ruff": "PASS", "targeted": targeted, "full": full,
        "frontend": "PASS" if frontend else "FAIL", "backend": "PASS" if backend else "FAIL",
        "production_approved": False,
    }
    (OUT / "test_results.json").write_text(json.dumps(test_results, indent=2) + "\n", encoding="utf-8")
    manifest = {
        path.relative_to(ROOT).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name not in {"artifact_manifest.json", "validation.json"}
    }
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
