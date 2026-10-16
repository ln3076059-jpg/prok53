"""Render evidence-backed V10 synthesis reports from immutable V9/V10 artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v10_research"


def _read(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _pct(value: float | None) -> str:
    return "NOT MEASURED" if value is None else f"{100 * value:.2f}%"


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    v9_path = "reports/v9_research/v9_diagnostics.json"
    gate_path = "reports/v9_research/phone_context_gate_ablation.json"
    replay_path = "reports/v9_research/phone_context_event_replay.json"
    review_path = "reports/v9_research/human_phone_review_summary.json"
    runtime_path = "reports/ai85_plus/runtime_benchmark.json"
    v9 = _read(v9_path)
    gate = _read(gate_path)
    replay = _read(replay_path)
    reviews = _read(review_path)
    runtime = _read(runtime_path)
    representation = _read("reports/v10_research/representation_probe.json")
    dataset = _read("reports/v10_research/dataset_manifest.json")
    models = _read("reports/v10_research/model_manifest.json")
    root_cause = _read("reports/v10_research/fn_root_cause.json")

    source_hashes = {
        path: sha256_file(ROOT / path)
        for path in (v9_path, gate_path, replay_path, review_path, runtime_path)
    }
    proposal = gate["proposal_summary"]
    floor = gate["thresholds"][0]
    _write(
        output / "06_PHONE_DETECTION_AND_ASSOCIATION.md",
        f"""# V10 phone detection and driver association

Evidence level: **LEVEL 1 diagnostic** on 13 repeatedly examined development FNs.

At the fixed 0.02 mining floor, the auxiliary detector produced {proposal['proposals']} proposals:
{proposal['geometry_contexts']['HANDHELD_USE']} were classified `HANDHELD_USE` and
{proposal['geometry_contexts']['MOUNTED_OR_STATIC']} were `MOUNTED_OR_STATIC`. Strict context gating removed
{floor['removed_proposals']} mounted/static candidates while retaining all {proposal['human_selected_proposals']}
human-selected handheld boxes. The remaining hard example is `{proposal['hard_example_case_ids'][0]}`.

This result measures proposal yield only. It does not measure detector AP, event recall, or independent
generalization. Raw proposal confidence remains separate from context admission. All 13 cases have
`UNVERIFIED_NO_OCCUPANT_TRACK`; the current geometry is a useful prior but cannot prove driver identity.
The association design therefore returns UNKNOWN when no driver track exists and blocks any claim that a
passenger or mounted device belongs to the driver.

No detector was fine-tuned: DMD rights are unaccepted, the reviewed lane is diagnostic-only, and training
on these selected FNs would leak evaluation cases. A future approved experiment needs full-frame/driver-ROI
coordinate round-trip tests, high-resolution border crops, person/seat tracks, hand-phone association, and
held-out negative scenarios for mounted camera, passenger phone, hand near ear, occlusion, and phone on lap.
""",
    )

    reps = {item["representation_id"]: item for item in representation["representations"]}
    motion = reps["D2_TEMPORAL_RESIDUAL"]
    baseline = reps["D0_FROZEN_RESNET_FULL_ROI"]
    _write(
        output / "08_MOTION_AND_VIDEO_ACTION_ABLATIONS.md",
        f"""# V10 motion and video-action ablations

The controlled CPU probe holds LOSO folds and the balanced linear head fixed. Temporal residual D2 raises
pooled clip Macro-F1 from {_pct(baseline['pooled']['macro_f1'])} to {_pct(motion['pooled']['macro_f1'])} and
SAFE recall from {_pct(baseline['pooled']['safe_recall'])} to {_pct(motion['pooled']['safe_recall'])}.
Subject silhouette falls from {baseline['separability']['subject_silhouette_cosine']:.4f} to
{motion['separability']['subject_silhouette_cosine']:.4f}, but action silhouette also falls from
{baseline['separability']['action_silhouette_cosine']:.4f} to
{motion['separability']['action_silhouette_cosine']:.4f}.

The worst D2 fold is `gZ-36`: Macro-F1 37.64%, SAFE recall 1.66%. D2 also uses a future adjacent frame, so it
is an offline non-causal representation. Event F1, onset latency, short-event recall, fragmentation, optical
flow, D3 CLIP, and D4 recurrent-head results are **NOT MEASURED** in V10. Downloads and larger training were
not justified on a no-CUDA host with unaccepted dataset rights. Decision: retain D2 only as a development
hypothesis; reject promotion.
""",
    )

    distributions = {
        subject: info["class_distribution"] for subject, info in dataset["subjects"].items()
    }
    motion_class = motion["pooled"]["per_class"]
    _write(
        output / "09_CLASS_IMBALANCE_AND_SAFE_DRIVING.md",
        f"""# V10 class imbalance and SAFE driving

The frozen three-subject cache contains 689 SAFE, 1,439 PHONE, and 497 OTHER coarse clips. The source
distribution by subject is preserved in `dataset_manifest.json`: `{json.dumps(distributions, sort_keys=True)}`.

D2 per-class F1 is SAFE {_pct(motion_class['SAFE_DRIVING']['f1'])}, PHONE
{_pct(motion_class['PHONE_USE']['f1'])}, and OTHER {_pct(motion_class['OTHER_ACTIVITY']['f1'])}. Pooled SAFE
recall is {_pct(motion_class['SAFE_DRIVING']['recall'])}, while `gZ-36` SAFE recall is only 1.66%, showing
subject-specific class collapse hidden by pooled results. Balanced class weights alone do not solve it.

No focal-loss, contrastive-loss, hard-negative-training, or calibration sweep was admitted. The 13 reviewed
FNs and their selected boxes remain diagnostic-only. Approved future data should explicitly include hand near
face, mirror adjustment, drinking, steering adjustment, mounted devices, passengers, and phone-like objects,
with group-disjoint splits and duplicate checks before loss-function ablations.
""",
    )

    naive = replay["aggregate"]["context_naive"]
    aware = replay["aggregate"]["context_aware"]
    _write(
        output / "10_MULTI_VIEW_FUSION_EXPERIMENTS.md",
        f"""# V10 multi-view fusion experiments

The only valid frozen comparison changes phone context while holding observations, temporal configuration,
ground truth, and matcher fixed. On two repeatedly used development subjects, context-naive replay is
TP={naive['true_positives']}, FP={naive['false_positives']}, FN={naive['false_negatives']}, F1={_pct(naive['f1'])},
FA/min={naive['false_alarms_per_minute']:.4f}. Context-aware replay is TP={aware['true_positives']},
FP={aware['false_positives']}, FN={aware['false_negatives']}, F1={_pct(aware['f1'])},
FA/min={aware['false_alarms_per_minute']:.4f}.

Measured delta: recovered TP=0, lost TP=0, removed FP=1, new FP=0, FN change=0. The historical artifact did
not persist event IDs for that removed FP, so ID-level traceability is **NOT AVAILABLE** and this result is
ineligible for promotion. `gC-14` has BODY only; missing FACE/HANDS is represented as missing information,
never negative evidence. G1/G2/G4 and a fully validated multi-view candidate remain unmeasured.
""",
    )

    temporal_rows = "\n".join(
        f"| {item['candidate']} | {item['true_positives']} | {item['false_positives']} | {item['false_negatives']} | {_pct(item['f1'])} | {item['false_alarms_per_minute']:.4f} |"
        for item in v9["fusion_ablation"]
        if item["candidate"] in {"AI85 next controlled binary", "T1", "T2", "T3", "T4", "T5"}
    )
    _write(
        output / "11_TEMPORAL_EVENT_EXPERIMENTS.md",
        f"""# V10 temporal event experiments

| Variant | TP | FP | FN | Event F1 | FA/min |
| --- | --- | --- | --- | --- | --- |
{temporal_rows}

T1 recovers one TP but adds nine FP relative to the controlled binary comparator; none of T1-T5 improves its
65.31% Event F1. Prior dense sampling still misses all four subsecond events. V10 therefore does not repeat a
fixed-density threshold sweep. D2 supplies a stronger clip-level motion hypothesis, but it is non-causal and
has no event predictions yet. Online onset/offset latency and fragmentation are **NOT MEASURED**. Ground truth,
event IoU, tolerance, and prediction-to-event semantics remain frozen.
""",
    )

    seatbelt = v9["seatbelt"]
    seatbelt_rows = "\n".join(
        f"| {item['id']} | {_pct(item['macro_f1'])} | {_pct(item['balanced_accuracy'])} | {_pct(item['uncertain_recall'])} | {_pct(item['false_unfastened_rate'])} | {_pct(item['abstention_rate'])} |"
        for item in seatbelt["frozen_variants"]
    )
    pending = seatbelt["pending_uncertain_review_manifest"]
    _write(
        output / "12_SEATBELT_RESEARCH_RESULTS.md",
        f"""# V10 seatbelt research results

| Frozen variant | Macro-F1 | Balanced accuracy | UNCERTAIN recall | False-unfastened | Abstention |
| --- | --- | --- | --- | --- | --- |
{seatbelt_rows}

Horizontal-flip TTA is the best measured development variant at {_pct(seatbelt['best_development_macro_f1'])};
it remains far below the 85% goal, and UNCERTAIN recall remains 4%. Resolution 320 degrades Macro-F1 and raises
false-unfastened rate, so increased resolution is rejected as a general remedy.

The pending review manifest contains {pending['samples']} samples, all PENDING, with zero automatic approved
training labels. Legacy UNCERTAIN errors have no local ROI evidence. Fine-tuning, Triplet Attention, torso-ROI
training, calibration error, selective risk, and coverage curves are therefore **BLOCKED/NOT MEASURED**.
Invisible or occluded belts remain UNCERTAIN and are never relabeled UNFASTENED.
""",
    )

    controlled = next(
        row for row in v9["fusion_ablation"] if row["candidate"] == "AI85 next controlled binary"
    )
    metrics = {
        "schema_version": 1,
        "evidence_scope": "DEVELOPMENT_ONLY",
        "protected_holdouts_accessed": False,
        "event_metrics": {
            "controlled_binary": controlled,
            "context_naive_two_subject": naive,
            "context_aware_two_subject": aware,
        },
        "clip_representation_metrics": {
            item["representation_id"]: {
                "pooled": item["pooled"],
                "folds": item["folds"],
                "separability": item["separability"],
            }
            for item in representation["representations"]
        },
        "seatbelt_metrics": seatbelt["frozen_variants"],
        "research_targets_achieved": False,
        "production_approved": False,
        "source_hashes": source_hashes,
    }
    (output / "metrics_comparison.json").write_text(
        json.dumps(metrics, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    _write(
        output / "13_METRICS_COMPARISON.md",
        f"""# V10 metrics comparison

| Scope | TP | FP | FN | Precision | Recall | F1 | FA/min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen controlled binary, 29 events | {controlled['true_positives']} | {controlled['false_positives']} | {controlled['false_negatives']} | {_pct(controlled['precision'])} | {_pct(controlled['recall'])} | {_pct(controlled['f1'])} | {controlled['false_alarms_per_minute']:.4f} |
| Context naive, two subjects | {naive['true_positives']} | {naive['false_positives']} | {naive['false_negatives']} | {_pct(naive['precision'])} | {_pct(naive['recall'])} | {_pct(naive['f1'])} | {naive['false_alarms_per_minute']:.4f} |
| Context aware, two subjects | {aware['true_positives']} | {aware['false_positives']} | {aware['false_negatives']} | {_pct(aware['precision'])} | {_pct(aware['recall'])} | {_pct(aware['f1'])} | {aware['false_alarms_per_minute']:.4f} |

The scopes differ and must not be ranked as one leaderboard. V10 D2 is a clip-level linear probe with
Macro-F1 {_pct(motion['pooled']['macro_f1'])}; it has no event metric. Best seatbelt development Macro-F1 is
{_pct(seatbelt['best_development_macro_f1'])}. No 85% research target is achieved.
""",
    )

    _write(
        output / "14_GENERALIZATION_VALIDATION.md",
        f"""# V10 generalization validation

Status: **{dataset['generalization_status']}**.

`gB-9`, `gC-14`, and `gZ-36` are repeatedly used development subjects. Protected `gZ-37` and `gE-28` were
not accessed. There are no new approved subjects and no independent holdout. DMD rights remain
`RIGHTS_ACCEPTED=false`, `CANONICAL_ELIGIBLE=false`; external upload and V10 fine-tuning are blocked.

The D2 LOSO folds are subject-disjoint but still LEVEL 2 because the three subjects and hypotheses have been
examined repeatedly. Confidence intervals and an independent claim are inappropriate. A valid next stage needs
provenance-approved subjects, deduplication, group-disjoint train/validation, a locked unseen test, then a
single frozen model/threshold/event-policy evaluation.
""",
    )

    pipe = runtime["full_backend_pipeline"]
    detector = runtime["detector"]
    temporal = runtime["temporal_head"]
    _write(
        output / "15_RUNTIME_BENCHMARK.md",
        f"""# V10 runtime benchmark

Evidence: inherited measured local CPU benchmark on the same AMD Ryzen 5 5500U class host, no CUDA.

| Component | p50 | p95 | Throughput/scope |
| --- | --- | --- | --- |
| Detector | {detector['p50_ms']:.2f} ms | {detector['p95_ms']:.2f} ms | {detector['count']} frames |
| Temporal head on cached embeddings | {temporal['p50_latency_ms']:.3f} ms | {temporal['p95_latency_ms']:.3f} ms | {temporal['clips_per_second']:.2f} clips/s |
| Full backend context | {pipe['per_context_latency']['p50_ms']:.2f} ms | {pipe['per_context_latency']['p95_ms']:.2f} ms | {pipe['processing_fps']:.3f} FPS |

Full pipeline memory peaked at {runtime['memory']['rss_peak_mb']:.1f} MiB. Detector, ROI, encoder, temporal,
fusion, and seatbelt were not all activated in one end-to-end call, so missing stage latencies remain **NOT
MEASURED**. Cached-feature throughput is not full-system FPS. GPU latency and memory are not reported because
no NVIDIA runtime is available.
""",
    )

    _write(
        output / "17_FINAL_AI_LEAD_REVIEW.md",
        f"""# V10 final AI lead review

Decision: **REJECT FOR PRODUCTION; CONTINUE CONTROLLED RESEARCH**.

The new measured result is a LEVEL 2 representation probe. Temporal residual D2 improves pooled clip
Macro-F1 to {_pct(motion['pooled']['macro_f1'])}, but worst-subject SAFE recall is 1.66%, action silhouette
falls, the representation is non-causal, and no event evaluation exists. The FN audit shows 12/13 reviewed
cases have usable auxiliary phone boxes, yet association is unverified in all 13 and dense action evidence
crosses the gate in only 4. These findings support representation, association, and temporal work; they do not
support deployment.

The controlled event comparator remains P={_pct(controlled['precision'])}, R={_pct(controlled['recall'])},
F1={_pct(controlled['f1'])}, FA/min={controlled['false_alarms_per_minute']:.4f}. Seatbelt best development
Macro-F1 is {_pct(seatbelt['best_development_macro_f1'])}. No independent holdout exists, rights are
unaccepted, and protected subjects were not opened. `production_approved=false`; no runtime config or active
checkpoint was changed.

Next admissible work is: obtain explicit data rights and new approved subjects; lock a group-disjoint holdout;
extract higher-resolution behavior/hand/face regions; build occupant association with UNKNOWN handling; turn D2
into a causal motion encoder; then evaluate event IDs, onset latency, short events, fragmentation, and all
negative contexts under a frozen policy. Seatbelt training waits for human-approved three-state labels.

Artifact inventory: {models['checkpoint_count']} local checkpoints hashed without deserialization. Source
hashes and machine-readable metrics are in `model_manifest.json`, `experiment_registry.json`, and
`metrics_comparison.json`.
""",
    )
    return {
        "reports_written": 10,
        "reviewed_phone_cases": reviews["summary"]["cases"],
        "root_cause_cases": root_cause["summary"]["cases"],
        "production_approved": False,
        "source_hashes": source_hashes,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
