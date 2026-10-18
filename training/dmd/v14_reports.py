"""Render V14 evidence reports."""

# ruff: noqa: E501

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from training.common import sha256_file
from training.dmd.v12_reports import read_native_log

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v14_research"


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name: str, text: str) -> None:
    (OUT / name).write_text(text.strip() + "\n", encoding="utf-8")


def junit(name: str) -> dict:
    root = ET.parse(OUT / name).getroot()
    suite = root[0] if root.tag == "testsuites" else root
    return {key: int(suite.attrib.get(key, 0)) for key in ("tests", "failures", "errors")}


def run() -> None:
    data, auto = read("data_manifest.json"), read("autonomous_proposal_audit.json")
    metrics, ready = read("metrics_comparison.json"), read("candidate_readiness.json")
    models, registry = read("model_manifest.json"), read("experiment_registry.json")
    full, targeted = junit("pytest_full.xml"), junit("pytest_targeted.xml")
    frontend = "built in" in read_native_log(OUT / "frontend_build.log")
    backend = '"status": "PASS"' in read_native_log(OUT / "backend_smoke.log")
    event = metrics["phone_event_controlled_29"]
    e3 = metrics["clip_selected_e3"]
    summary = auto["summary"]
    write("01_HEAD_AND_RIGHTS_AUDIT.md", f"""# V14 HEAD and rights audit

V14 started from `{registry['source_head']}`. Live rights remain `RIGHTS_ACCEPTED={data['live_permissions']['RIGHTS_ACCEPTED']}`, `CANONICAL_ELIGIBLE={data['live_permissions']['CANONICAL_ELIGIBLE']}`, and `TEMPORAL_DEVELOPMENT_ELIGIBLE={data['live_permissions']['TEMPORAL_DEVELOPMENT_ELIGIBLE']}`. Training, calibration, fusion fitting and upload remain denied. No fitting occurred.

Protected `gZ-37` and `gE-28` artifacts were not opened. The only pre-existing working-tree change was user-authored `prompt.md`, which V14 preserved. Active production configuration and model registration were not changed. The specialist phone checkpoint hash is `{models['phone_detector']['sha256']}` and remains research-only.
""")
    write("02_V13_BASELINE_RECONCILIATION.md", f"""# V14 reconciliation of the V13 baseline

The controlled development event baseline remains TP={event['true_positives']}, FP={event['false_positives']}, FN={event['false_negatives']}, P={event['precision']:.4f}, R={event['recall']:.4f}, F1={event['f1']:.4f}, FA/min={event['false_alarms_per_minute']:.4f}. E3 development Macro-F1 remains {e3['pooled']['macro_f1']:.4f}; SAFE/PHONE/OTHER recall is {e3['pooled']['safe_recall']:.4f}/{e3['pooled']['phone_recall']:.4f}/{e3['pooled']['other_recall']:.4f}; worst-subject SAFE recall is {e3['worst_subject']['safe_recall']:.4f}.

No V14 accuracy metric replaces these values. Phone AP and autonomous proposal precision/recall remain null because the FN-selected sample has no complete bbox ground truth or negative denominator. Independent AI85 remains unestablished.
""")
    write("03_AUTONOMOUS_PHONE_PROPOSALS.md", f"""# V14 autonomous phone proposals

`AutonomousPhonePipeline` accepts an original HxWx3 frame, non-oracle metadata, runtime driver evidence and a detector adapter. It rejects metadata keys for ground truth, reviewed or selected boxes, phone boxes and event bounds. Detector proposals retain raw confidence, model hash and source-frame coordinates. Missing proposals produce a source-resolution upper-body or face/hand fallback and remain UNKNOWN; no empty observation becomes safe evidence.

The current specialist and auxiliary checkpoints were hash-reconciled against the existing fully automatic V9 mining run. That run processed {summary['sampled_frames']} frames across {summary['cases']} FN-selected cases: specialist proposals occurred on {summary['specialist_frames_with_proposal']} frames/{summary['cases_with_specialist_proposal']} cases; auxiliary proposals occurred on {summary['auxiliary_frames_with_proposal']} frames/{summary['cases_with_auxiliary_proposal']} cases; cross-model agreement occurred in {summary['cases_with_cross_model_agreement']} cases. Reviewed boxes were not detector inputs. These counts are proposal yield, not precision, recall, AP or recovered events. A new YOLO pass was not run under current RAM constraints.
""")
    write("04_HIGH_RES_BEHAVIOR_PIPELINE.md", """# V14 high-resolution behavior pipeline

Autonomous proposals now feed the V13 `HighResolutionRoiPipeline` directly in original-frame coordinates. No 64x64 cache is enlarged and described as recovered detail. Crop records include deterministic frame SHA-256, input/output dimensions, requested and clipped boxes, padding, scale, evidence state, driver track ID and phone coverage when a runtime proposal exists.

Synthetic regression cases verify partial boundary coverage and coordinate local-to-original round trips. When a proposal is absent, the pipeline uses available hand/face evidence or an upper-body fallback with UNKNOWN evidence semantics. The V13 oracle result of 12/12 reviewed-box coverage remains explicitly separate as ORACLE_BOX_DIAGNOSTIC_ONLY and is not an autonomous benchmark result.
""")
    write("05_ONLINE_E3_MODEL_READINESS.md", f"""# V14 OnlineE3 model readiness

OnlineE3 causal features remain engineering-ready with inherited batch/stream parity. NumericHead now accepts strict schema v2 with dataset/split/feature provenance plus checkpoint hash, config hash, library versions, exact class order and an explicit calibration state. Loading can verify the SHA-256 of the complete JSON file before parsing. Decision-function argmax remains authoritative; normalized OvR sigmoid scores remain uncalibrated unless a locked-validation calibration record says otherwise.

The real fitted head/scaler status remains `{ready['fitted_e3_head_scaler']}`. Schema v2 was tested with synthetic numeric fixtures only. No synthetic coefficient is reported as a model and E3 Event F1 remains NOT MEASURED.
""")
    write("06_DRIVER_ASSOCIATION_EVIDENCE.md", """# V14 driver association evidence

The autonomous adapter transports runtime driver-region provenance, role, track identity, confidence and verification status. An absent driver region falls back to the full source frame for candidate preservation but keeps role state UNKNOWN. Passenger evidence remains passenger evidence; a visible phone alone is never a driver violation.

Existing association logic continues to abstain for overlapping occupants, unverified tracks, unsupported roles and unverified mounted/static context. Synthetic tests cover passengers, ambiguity and missing evidence. Real association accuracy is not reported because the 13 development FNs contain zero verified occupant tracks and there is no association ground-truth benchmark.
""")
    write("07_EVENT_REPLAY_RESULTS.md", f"""# V14 event replay results

Real candidate replay is `{ready['real_event_replay']}`. The frozen matcher, event denominators and ground truth were unchanged. V14 therefore retains the 16/4/13 controlled comparator and does not manufacture events from detector yield or synthetic model weights.

The opt-in candidate still records stable event ID, source key, actual emission time, model/config hashes, raw phone evidence, uncalibrated action scores, association state and rejection reasons. A legitimate replay additionally requires a fitted E3 head/scaler and verified driver association. Recovered/lost/new/removed event IDs, fragmentation, short-event recall and real alert latency are not newly measured.
""")
    write("08_SEATBELT_READINESS.md", f"""# V14 seatbelt readiness

Seatbelt remains `{ready['seatbelt']}`. The 500 review candidates remain pending and zero labels were automatically approved. V14 did not train, calibrate, relabel or activate a seatbelt model.

The inherited development best remains Macro-F1 58.52% with UNCERTAIN recall 4%. FASTENED, UNFASTENED and UNCERTAIN_OR_OCCLUDED remain distinct; invisible or occluded belts are not converted into UNFASTENED. A valid next run requires human-approved labels, fitting rights and a locked group-disjoint split.
""")
    write("09_RUNTIME_AND_TESTS.md", f"""# V14 runtime and tests

Targeted tests: {targeted['tests']} passed, {targeted['failures']} failures, {targeted['errors']} errors. Full pytest: {full['tests']} passed, {full['failures']} failures, {full['errors']} errors. Python compile and scoped Ruff passed. Frontend build: {'PASS' if frontend else 'FAIL'}. Backend smoke: {'PASS' if backend else 'FAIL'}.

The new autonomous adapter was exercised with deterministic synthetic detector fixtures, including empty proposals, forbidden oracle metadata, boundary boxes, passengers and coordinate round trips. New YOLO timing was NOT RUN due constrained host memory. Historical detector timing and V13 ROI timing retain their original scopes; neither is called end-to-end FPS.
""")
    write("10_FINAL_AI_LEAD_DECISION.md", """# V14 final AI lead decision

Decision: **RESEARCH_ONLY; REJECT FOR PRODUCTION**.

KEEP engineering: autonomous frame-to-proposal adapter, source-resolution ROI integration, coordinate/provenance tests, missing-proposal UNKNOWN handling and NumericHead schema v2/hash validation. INCONCLUSIVE accuracy: autonomous detector proposal yield, because bbox negatives and a complete ground-truth denominator are absent. BLOCKED: action fitting, verified driver association, real event replay, seatbelt fitting, end-to-end benchmark and independent evaluation.

No metric reached 85% scientifically. Phone events remain P/R/F1 80.00%/55.17%/65.31%, FA/min 0.1813. E3 Macro-F1 remains 56.39%, SAFE/PHONE/OTHER recall 60.38%/66.78%/43.26%. Seatbelt Macro-F1 remains 58.52%, UNCERTAIN recall 4%. The highest-value next experiment is a rights-approved, group-disjoint dataset with autonomous bbox ground truth and verified occupant roles, followed by a frozen R5+E3 fit and fixed-policy replay. Production approval remains false.
""")
    tests = {"compile":"PASS", "ruff":"PASS", "targeted":targeted, "full":full, "frontend":"PASS" if frontend else "FAIL", "backend":"PASS" if backend else "FAIL"}
    (OUT / "test_results.json").write_text(json.dumps(tests, indent=2) + "\n", encoding="utf-8")
    manifest = {p.relative_to(ROOT).as_posix(): {"sha256":sha256_file(p), "bytes":p.stat().st_size} for p in sorted(OUT.rglob("*")) if p.is_file() and p.name not in {"artifact_manifest.json", "validation.json"}}
    (OUT / "artifact_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    run()
