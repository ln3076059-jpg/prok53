"""Render evidence-backed V11 reports and machine-readable research artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from training.common import sha256_file

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v11_research"


def _read(relative: str) -> Any:
    return json.loads((ROOT / relative).read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _pct(value: float | None) -> str:
    return "NOT MEASURED" if value is None else f"{100 * value:.2f}%"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def _rep_rows(data: dict[str, Any]) -> str:
    return _table(
        ["ID", "Macro-F1", "Balanced acc.", "SAFE R", "PHONE R", "OTHER R", "Worst SAFE", "Worst Macro-F1", "Decision"],
        [
            [
                item["representation_id"],
                _pct(item["pooled"]["macro_f1"]),
                _pct(item["pooled"]["balanced_accuracy"]),
                _pct(item["pooled"]["safe_recall"]),
                _pct(item["pooled"]["phone_recall"]),
                _pct(item["pooled"]["other_recall"]),
                _pct(item["worst_subject"]["safe_recall"]),
                _pct(item["worst_subject"]["macro_f1"]),
                "SELECTED" if item["representation_id"] == data["selected_candidate"] else "REJECT",
            ]
            for item in data["representations"]
        ],
    )


def run(output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.mkdir(parents=True, exist_ok=True)
    repo_env = _read("reports/v11_research/repository_environment.json")
    dataset = _read("reports/v11_research/dataset_manifest.json")
    models = _read("reports/v11_research/model_manifest.json")
    literature = _read("reports/v11_research/literature_review.json")
    fn = _read("reports/v11_research/fn_root_cause.json")
    reps = _read("reports/v11_research/representation_experiments.json")
    gen = _read("reports/v11_research/generalization_experiments.json")
    v9 = _read("reports/v9_research/v9_diagnostics.json")
    replay = _read("reports/v9_research/phone_context_event_replay.json")
    runtime = _read("reports/ai85_plus/runtime_benchmark.json")

    selected = next(row for row in gen["representations"] if row["representation_id"] == gen["selected_candidate"])
    frozen = next(row for row in reps["representations"] if row["representation_id"] == "V11-A_FROZEN_RESNET")
    raw_causal = next(row for row in gen["representations"] if row["representation_id"] == "E0_RAW_CAUSAL_MULTISCALE")
    controlled = next(row for row in v9["fusion_ablation"] if row["candidate"] == "AI85 next controlled binary")
    naive = replay["aggregate"]["context_naive"]
    aware = replay["aggregate"]["context_aware"]

    confusion = {
        "schema_version": 1,
        "class_order": ["SAFE_DRIVING", "PHONE_USE", "OTHER_ACTIVITY"],
        "representation_experiments": {
            row["representation_id"]: {
                "pooled": row["pooled"]["confusion_matrix"],
                "per_subject": {fold["heldout_subject"]: fold["confusion_matrix"] for fold in row["folds"]},
            }
            for row in reps["representations"]
        },
        "generalization_experiments": {
            row["representation_id"]: {
                "pooled": row["pooled"]["confusion_matrix"],
                "per_subject": {fold["heldout_subject"]: fold["confusion_matrix"] for fold in row["folds"]},
            }
            for row in gen["representations"]
        },
    }
    per_subject = {
        "schema_version": 1,
        "evidence_level": "LEVEL_2_REPEATED_DEVELOPMENT_LOSO",
        "selected_candidate": gen["selected_candidate"],
        "subjects": {fold["heldout_subject"]: fold for fold in selected["folds"]},
        "independent_validation": False,
    }
    event_evidence = {
        "schema_version": 1,
        "frozen_controlled_binary": controlled,
        "inherited_context_replay": replay["aggregate"],
        "recovered_tp_ids": [],
        "lost_tp_ids": [],
        "new_fp_ids": [],
        "removed_fp_ids": "NOT_PERSISTED_BY_INHERITED_ARTIFACT",
        "v11_new_event_predictions": False,
        "ground_truth_changed": False,
        "matcher_changed": False,
        "warning": "Clip classification and auxiliary proposal yield are not event recovery.",
    }
    _write_json(output / "confusion_matrices.json", confusion)
    _write_json(output / "per_subject_results.json", per_subject)
    _write_json(output / "event_match_evidence.json", event_evidence)

    paper_rows = [
        [
            row["title"], row["authors"], row["year_venue"], row["dataset"], row["metrics_and_definition"],
            f"{row['code']} / {row['checkpoint']}", row["license"], row["transferable_method"], row["limitations"],
        ]
        for row in literature["papers"]
    ]
    _write(
        output / "01_LITERATURE_REVIEW.md",
        f"""# V11 literature review and reproducibility

Paper metrics below remain paper-level evidence and are not Roadwatch results. V11 did not download weights or reproduce a full external architecture because DMD rights are pending, the host has no NVIDIA runtime, and RAM is limited.

{_table(['Title', 'Authors', 'Venue', 'Dataset/protocol', 'Metric scope', 'Code/checkpoint', 'License', 'Transferable idea', 'Roadwatch limit'], paper_rows)}

Applied ideas are deliberately smaller than the source systems: causal multi-scale history from DualScope/ActionFormer, behavior-region pooling, missing-view uncertainty, worst-subject selection, and feature-statistic normalization inspired by domain-generalization work. The experiment registry distinguishes these adaptations from reproduction. DriveCLIP remains blocked because the checkpoint lineage, download size, and memory budget were not cleared. Group DRO was not run with only three repeatedly studied subjects. Seatbelt attention remains blocked until three-state labels and rights are approved.
""",
    )

    repo = repo_env["repository"]
    env = repo_env["environment"]
    _write(
        output / "02_REPOSITORY_AUDIT.md",
        f"""# V11 repository and environment audit

- Branch: `{repo['branch']}`; audit HEAD: `{repo['head']}`.
- Upstream at audit: `{repo['upstream']}`; synchronized before V11 work: `{repo['head_matches_upstream_at_audit']}`.
- User-owned `PRODUCT.md` change was preserved and excluded from V11-generated source changes.
- CPU: `{env['cpu']}`, {env['physical_cores']} physical and {env['logical_cores']} logical cores.
- RAM: {env['ram_total_bytes'] / 2**30:.2f} GiB total, {env['ram_available_bytes'] / 2**30:.2f} GiB available at P0 capture.
- GPU: `{env['gpu']['status']}`; CUDA available: `{env['gpu']['cuda_gpu_available']}`.
- Python: `{env['python']}`; package inventory: `{env['packages']}`.
- Local checkpoints: {models['checkpoint_count']} files, {models['total_bytes'] / 2**20:.1f} MiB, byte-hashed without deserialization.
- V10-to-V11 checkpoint hash reconciliation: `{models['v10_hash_reconciliation']['status']}` with {len(models['v10_hash_reconciliation']['mismatches'])} mismatches.

This environment supports audits, deterministic CPU feature probes, and unit tests. It does not support honest GPU training or a full ActionFormer/large CLIP reproduction. No active runtime config, production checkpoint, ground truth, event matcher, or protected holdout was changed. No automatic push or merge to `main` was performed.
""",
    )

    _write(
        output / "03_DATA_GOVERNANCE.md",
        f"""# V11 data governance

Status: **{dataset['generalization_status']}**. `RIGHTS_ACCEPTED={str(dataset['rights_accepted']).lower()}` and `CANONICAL_ELIGIBLE={str(dataset['canonical_eligible']).lower()}`. Local temporal diagnostics are allowed, while new DMD fine-tuning and external uploads are blocked. Development subjects are `{', '.join(dataset['development_subjects'])}`; each has been repeatedly inspected, so all new results are LEVEL 2 at most.

Protected subjects `{', '.join(dataset['protected_holdouts'])}` were neither enumerated nor opened by V11. They were not used for training, threshold choice, model selection, or reporting. There are no new approved subjects and no independent holdout. The reviewed 13-phone-FN lane remains diagnostic-only, with 12 valid reviewed auxiliary boxes and one visible-phone hard example without a valid box. Those selected examples were not converted into training labels.

Missing BODY/FACE/HANDS evidence is represented as UNKNOWN. An occluded belt remains `UNCERTAIN_OR_OCCLUDED`; it is never automatically relabeled `UNFASTENED`. Before any future training, the project must verify permission, provenance, human label approval, subject/camera groups, deduplication, and a locked split.
""",
    )

    _write(
        output / "04_FROZEN_BASELINE.md",
        f"""# V11 frozen baseline

The frozen event comparator is unchanged: TP={controlled['true_positives']}, FP={controlled['false_positives']}, FN={controlled['false_negatives']}, precision={_pct(controlled['precision'])}, recall={_pct(controlled['recall'])}, Event F1={_pct(controlled['f1'])}, and FA/min={controlled['false_alarms_per_minute']:.4f}. Ground truth and matching policy are unchanged.

The V11-A clip baseline reuses the local frozen ResNet representation with the pre-registered LOSO protocol. It obtains Macro-F1 {_pct(frozen['pooled']['macro_f1'])}, balanced accuracy {_pct(frozen['pooled']['balanced_accuracy'])}, SAFE recall {_pct(frozen['pooled']['safe_recall'])}, and worst-subject SAFE recall {_pct(frozen['worst_subject']['safe_recall'])}. Its subject silhouette is {frozen['separability']['subject_silhouette_cosine']:.4f}, far larger than its action silhouette {frozen['separability']['action_silhouette_cosine']:.4f}, confirming strong subject/cabin bias.

These are two distinct tasks: the event comparator cannot be compared directly with clip Macro-F1. V11-A provides a controlled clip baseline only. It neither replaces the active model nor claims phone AP, mAP, event recovery, independent generalization, or production readiness.
""",
    )

    fn_rows = []
    for case in fn["cases"]:
        obj = case["object_detection"]
        action = case["action_evidence"]
        fn_rows.append([
            case["event_id"], f"{case['timestamp']['start_seconds']:.2f}-{case['timestamp']['end_seconds']:.2f}",
            case["ground_truth_action"].split("/")[-1], obj["specialist_raw_max_confidence"],
            f"{obj['human_selected_auxiliary_confidence']:.3f}" if obj["human_selected_auxiliary_confidence"] is not None else "NO VALID BOX",
            f"{action['T1_dense_max']:.3f}", case["driver_association"]["state"], ", ".join(case["failure_stages"]),
        ])
    _write(
        output / "05_FALSE_NEGATIVE_ROOT_CAUSE.md",
        f"""# V11 false-negative root cause

Evidence scope: authenticated review of the same 13 development false negatives. Each machine-readable case includes timestamp/frame range, action, visibility, raw specialist and auxiliary confidence, driver-ROI and original-frame box coordinates, hand/face proximity, the complete frozen action-probability timeline, gate state, missed interval, and review receipt.

{_table(['Event', 'Time (s)', 'Action', 'Specialist max', 'Selected aux.', 'T1 max', 'Association', 'Failure stages'], fn_rows)}

All {fn['summary']['phone_visible']}/13 have a confirmed visible phone and confirmed use. Twelve have a human-selected auxiliary candidate; `gB-9-phone-06` has none. All 13 lack a verified occupant track. Only {fn['summary']['T1_reaches_gate']}/13 reach the frozen dense action gate, so the dominant path combines specialist detector miss, candidate-admission failure, unknown driver association, and weak action evidence. A reviewed box is proposal yield, not a recovered event; V11 claims zero newly recovered events.
""",
    )

    roi_example = reps["roi_provenance_example"]
    _write(
        output / "06_BEHAVIOR_ROI_EXPERIMENTS.md",
        f"""# V11 behavior ROI experiments

`training/dmd/behavior_roi.py` implements C0 full driver, C1 upper body, C2 hand-phone, C3 face-hand, C4 multi-region, and C5 dynamically padded crops. Every crop returns original coordinates, requested/crop coordinates, scale, replicated padding, confidence, camera, track ID or UNKNOWN, evidence state, and reason codes. The recorded example is `{json.dumps(roi_example, ensure_ascii=False)}`.

V11-D fixed multi-ROI appearance reaches Macro-F1 {_pct(next(x for x in reps['representations'] if x['representation_id'] == 'V11-D_BEHAVIOR_MULTI_ROI')['pooled']['macro_f1'])}, but its worst-subject Macro-F1 is {_pct(next(x for x in reps['representations'] if x['representation_id'] == 'V11-D_BEHAVIOR_MULTI_ROI')['worst_subject']['macro_f1'])}. V11-E ROI plus causal motion raises pooled Macro-F1 to {_pct(next(x for x in reps['representations'] if x['representation_id'] == 'V11-E_ROI_PLUS_CAUSAL')['pooled']['macro_f1'])}, while worst-subject SAFE recall falls to {_pct(next(x for x in reps['representations'] if x['representation_id'] == 'V11-E_ROI_PLUS_CAUSAL')['worst_subject']['safe_recall'])}; it is rejected by the pre-registered stability gate.

Boundary replication prevents a phone at the driver-ROI edge from being cropped away. Mounted/static and passenger cases are handled by association logic rather than by appearance alone. This probe uses fixed regions on a 64×64 cache, so it is a representation diagnostic, not a measured high-resolution phone detector or AP experiment.
""",
    )

    _write(
        output / "07_CAUSAL_MOTION_EXPERIMENTS.md",
        f"""# V11 causal motion experiments

`CausalMotionEncoder` uses only the current feature and historical lags. The pre-registered lags 1, 3, and 7 correspond to about 0.504, 1.512, and 3.528 seconds in the frozen cache. Tests prove that changing future rows cannot change prior outputs. Causal running standardization also computes every row from prior history only.

{_rep_rows(reps)}

Lag-1 motion improves pooled Macro-F1 over the frozen appearance baseline, and multi-scale causal motion improves it further to {_pct(next(x for x in reps['representations'] if x['representation_id'] == 'V11-C_CAUSAL_MULTISCALE')['pooled']['macro_f1'])}. However, its worst-subject SAFE recall is only {_pct(next(x for x in reps['representations'] if x['representation_id'] == 'V11-C_CAUSAL_MULTISCALE')['worst_subject']['safe_recall'])}. Every initial representation candidate therefore failed the worst-subject stop rule. No future frame, bidirectional temporal head, or ground-truth boundary was used. No event predictions were generated from these clip scores.
""",
    )

    _write(
        output / "08_CLIP_AND_DUALSCOPE_COMPARISON.md",
        f"""# V11 CLIP and DualScope comparison

DualScope directly motivates critical behavior regions, dynamic differences, and fine-to-coarse time scales. V11 implements a small causal analogue with fixed region pooling and historical differences; it does not claim to reproduce DualScope's architecture, distillation, datasets, or published metrics. The initial ROI-plus-motion variant obtains pooled Macro-F1 {_pct(next(x for x in reps['representations'] if x['representation_id'] == 'V11-E_ROI_PLUS_CAUSAL')['pooled']['macro_f1'])} but collapses on the worst subject, so it is rejected.

DriveCLIP motivates frozen semantic embeddings and task adapters. D4 was not run: no suitable locally verified checkpoint was admitted, ViT-L/14 would add significant download/RAM cost, and dataset rights forbid moving DMD media to an external service. Reporting a CLIP number from a different split or paper would be scientifically invalid.

The accepted development hypothesis is therefore the small E3 normalization-plus-causal representation, which uses existing frozen features and CPU compute. It offers an inspectable causal path and better subject balance, but cannot establish the richer semantic transfer claimed by CLIP or the full spatiotemporal design of DualScope.
""",
    )

    fold_rows = [[fold["heldout_subject"], _pct(fold["macro_f1"]), _pct(fold["safe_recall"]), _pct(fold["phone_recall"]), _pct(fold["other_recall"])] for fold in selected["folds"]]
    _write(
        output / "09_SUBJECT_GENERALIZATION.md",
        f"""# V11 subject generalization

{_rep_rows(gen)}

Selected E3 per-subject results:

{_table(['Held-out subject', 'Macro-F1', 'SAFE recall', 'PHONE recall', 'OTHER recall'], fold_rows)}

E3 combines spatial residual normalization with past-only running normalization. Relative to raw causal E0, pooled Macro-F1 changes from {_pct(raw_causal['pooled']['macro_f1'])} to {_pct(selected['pooled']['macro_f1'])}, worst-subject SAFE recall from {_pct(raw_causal['worst_subject']['safe_recall'])} to {_pct(selected['worst_subject']['safe_recall'])}, and subject silhouette from {raw_causal['separability']['subject_silhouette_cosine']:.4f} to {selected['separability']['subject_silhouette_cosine']:.4f}. Runtime subject identity is not an input.

E3 passes the pre-registered development stability gate and is kept as a hypothesis. It is not promoted: three subjects are too few, all have been repeatedly examined, no confidence interval would restore independence, and no locked holdout exists. E4 balanced subject-class sampling regresses worst-subject SAFE recall, showing that balancing alone does not solve cabin/subject shift.
""",
    )

    _write(
        output / "10_DRIVER_PHONE_ASSOCIATION.md",
        """# V11 driver-phone association

`DriverPhoneAssociation` returns exactly four states: `DRIVER_ASSOCIATED`, `PASSENGER_ASSOCIATED`, `MOUNTED_OR_STATIC`, or `UNKNOWN`, together with confidence, evidence, and reason codes. Verified mounted/static context takes precedence. A verified passenger track cannot create a driver event. Proximity alone never assigns the nearest person as driver.

For the real 13-FN audit, the output is UNKNOWN in 13/13 cases because none has a verified occupant track. That result is intentional uncertainty, not a failure hidden as negative evidence. The human-reviewed auxiliary boxes remain useful for object/geometry diagnostics, while driver attribution requires person/seat tracking or verified calibration.

Synthetic regression tests cover verified driver containment, passenger phone use, mounted cabin equipment, unverified tracks, and missing evidence. These tests establish code semantics only; they do not measure association accuracy on DMD. No event precision gain is claimed until association outputs are replayed with persisted event IDs under the frozen matcher.
""",
    )

    _write(
        output / "11_FUSION_ABLATION.md",
        f"""# V11 fusion ablation

The only available frozen event replay is inherited from V9 and covers two development subjects. Context-naive fusion records TP={naive['true_positives']}, FP={naive['false_positives']}, FN={naive['false_negatives']}, F1={_pct(naive['f1'])}, FA/min={naive['false_alarms_per_minute']:.4f}. Context-aware fusion records TP={aware['true_positives']}, FP={aware['false_positives']}, FN={aware['false_negatives']}, F1={_pct(aware['f1'])}, FA/min={aware['false_alarms_per_minute']:.4f}.

Measured aggregate delta is recovered TP=0, lost TP=0, removed FP=1, new FP=0, and FN change=0. The inherited artifact did not retain the removed FP event ID, so ID-level evidence is NOT AVAILABLE. G1-G6 were not rerun because the V11 clip candidate does not yet produce calibrated event evidence and real occupant association remains UNKNOWN.

Future fusion must persist raw object, action, geometry, association, fused confidence, rejection reasons, and source views per event. Missing views remain UNKNOWN. Passenger and mounted devices cannot create driver-phone alerts. V11 does not use a new hard gate or claim event improvement from clip metrics.
""",
    )

    _write(
        output / "12_TEMPORAL_LOCALIZATION.md",
        """# V11 temporal localization

ActionFormer supports multi-scale temporal features and direct boundary regression, but a full reproduction was not admitted: there is no approved large event-training set, no CUDA GPU, and current rights do not permit external training. V11 only establishes causal clip features as input for a future lightweight boundary model.

No H1-H5 event prediction artifact was generated. Therefore onset error, offset error, fragmentation, duplicate alerts, duration-bin recall, and online alarm latency are NOT MEASURED for the V11 representation. The frozen event comparator and its matcher remain unchanged. Prior T1-T5 temporal experiments did not improve Event F1, and dense sampling alone missed four subsecond events.

A valid next experiment should freeze causal inputs, learn separate start/end evidence on approved training subjects, rank candidates without ground-truth extension, and emit complete event IDs. Evaluation must retain the existing matcher and report events below one second, 1-3 seconds, 3-10 seconds, over 10 seconds, and full-pipeline runtime. Until that artifact exists, no clip score is described as temporal event recovery.
""",
    )

    seatbelt = v9["seatbelt"]
    _write(
        output / "13_SEATBELT_EXPERIMENTS.md",
        f"""# V11 seatbelt experiments

The frozen seatbelt variants remain the only measured evidence. Best development Macro-F1 is {_pct(seatbelt['best_development_macro_f1'])}; UNCERTAIN recall remains 4% in the inherited evaluation. This is far below the 85% research target and does not justify production use.

The 500-candidate annotation manifest remains pending. Model-assisted suggestions are not ground truth, and V11 creates zero automatic approved labels. Fine-tuning, torso/upper-torso/belt-line ROI comparison, Triplet Attention, visibility-state multitask learning, calibration error, selective risk, and coverage-adjusted risk are therefore BLOCKED/NOT MEASURED under the data-rights and human-approval gates.

The state contract remains `FASTENED`, `UNFASTENED`, and `UNCERTAIN_OR_OCCLUDED`. Invisible or occluded belt evidence cannot become `UNFASTENED`. A future experiment must report per-class F1, Macro-F1, balanced accuracy, false-unfastened rate, abstention rate, and risk at coverage; predicting UNCERTAIN for nearly every sample will not count as success.
""",
    )

    metrics = {
        "schema_version": 1,
        "scope": "DEVELOPMENT_ONLY",
        "event": {"controlled_binary": controlled, "context_naive": naive, "context_aware": aware},
        "clip": {"frozen": frozen["pooled"], "selected_e3": selected["pooled"], "selected_e3_folds": selected["folds"]},
        "seatbelt": {"best_development_macro_f1": seatbelt["best_development_macro_f1"], "variants": seatbelt["frozen_variants"]},
        "phone_ap50": None,
        "overall_map50": None,
        "overall_map50_95": None,
        "independent_ai85_established": False,
        "production_approved": False,
    }
    _write_json(output / "metric_comparison.json", metrics)
    _write(
        output / "14_EVENT_METRICS.md",
        f"""# V11 event metrics

| Scope | TP | FP | FN | Precision | Recall | F1 | FA/min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen controlled binary, 29 events | {controlled['true_positives']} | {controlled['false_positives']} | {controlled['false_negatives']} | {_pct(controlled['precision'])} | {_pct(controlled['recall'])} | {_pct(controlled['f1'])} | {controlled['false_alarms_per_minute']:.4f} |
| Inherited context-naive replay, two subjects | {naive['true_positives']} | {naive['false_positives']} | {naive['false_negatives']} | {_pct(naive['precision'])} | {_pct(naive['recall'])} | {_pct(naive['f1'])} | {naive['false_alarms_per_minute']:.4f} |
| Inherited context-aware replay, two subjects | {aware['true_positives']} | {aware['false_positives']} | {aware['false_negatives']} | {_pct(aware['precision'])} | {_pct(aware['recall'])} | {_pct(aware['f1'])} | {aware['false_alarms_per_minute']:.4f} |

V11 E3 is a clip classifier with Macro-F1 {_pct(selected['pooled']['macro_f1'])}; it has no Event F1. Phone AP50, overall mAP50, and mAP50-95 are NOT MEASURED because the reviewed FN lane is not a complete detection test set. No 85% target is met. Event match evidence explicitly records zero V11 recovered/lost/new event IDs and preserves the frozen ground truth and matcher.
""",
    )

    pipe = runtime["full_backend_pipeline"]
    detector = runtime["detector"]
    _write(
        output / "15_RUNTIME_RESULTS.md",
        f"""# V11 runtime results

The inherited same-host CPU cabin-video benchmark reports detector p50 {detector['p50_ms']:.2f} ms and p95 {detector['p95_ms']:.2f} ms over {detector['count']} frames. Full backend context latency is p50 {pipe['per_context_latency']['p50_ms']:.2f} ms and p95 {pipe['per_context_latency']['p95_ms']:.2f} ms, with {pipe['processing_fps']:.3f} processing FPS and real-time factor {pipe['real_time_factor']:.3f}. Peak RSS was {runtime['memory']['rss_peak_mb']:.1f} MiB.

The V11 representation family took {reps['runtime']['total_seconds']:.2f} seconds and the generalization family took {gen['runtime_seconds']:.2f} seconds on CPU, including all three LOSO folds in each planned variant. Fold-specific fit/predict times are stored with each prediction artifact.

Cached-feature throughput is not full-pipeline FPS. V11 did not rerun a new end-to-end detector+ROI+encoder+temporal+fusion pipeline, so incremental ROI and causal-motion latency inside production is NOT MEASURED. GPU latency and VRAM are not reported because no NVIDIA runtime is present.
""",
    )

    # Report 16 is finalized by v11_test_report after test execution.
    _write(
        output / "16_TEST_RESULTS.md",
        """# V11 test results

Status: **PENDING FINAL TEST RUN**.

The final test report is generated after Python compilation, scoped Ruff, targeted V11 pytest, full pytest, frontend production build, backend OpenAPI smoke, metric reconciliation, prediction-hash verification, rights enforcement, and protected-holdout guards. This placeholder is intentionally replaced by `training.dmd.v11_test_report`; it is not a PASS claim.

Current unit coverage includes causal future-invariance, past-only running normalization, ROI transforms and boundary padding, missing evidence, mounted/static context, passenger association, unverified occupants, protected-subject rejection, deterministic metrics, and closed rights gates. No test is deleted or relaxed to obtain a successful result.
""",
    )

    _write(
        output / "17_FINAL_AI_RESEARCH_REVIEW.md",
        f"""# V11 final AI research review

Decision: **RESEARCH_ONLY; REJECT FOR PRODUCTION**.

V11 adds reusable behavior ROI, causal motion, and driver-phone association modules. The initial representation set failed its worst-subject gate. A follow-up, pre-declared normalization study selected E3: development clip Macro-F1 {_pct(selected['pooled']['macro_f1'])}, SAFE recall {_pct(selected['pooled']['safe_recall'])}, PHONE recall {_pct(selected['pooled']['phone_recall'])}, worst-subject Macro-F1 {_pct(selected['worst_subject']['macro_f1'])}, and worst-subject SAFE recall {_pct(selected['worst_subject']['safe_recall'])}. Subject silhouette falls to {selected['separability']['subject_silhouette_cosine']:.4f}. This supports continued study of causal subject-normalized features.

It does not establish the requested system targets. The controlled event comparator remains P={_pct(controlled['precision'])}, R={_pct(controlled['recall'])}, F1={_pct(controlled['f1'])}, FA/min={controlled['false_alarms_per_minute']:.4f}; no new V11 Event F1 exists. Seatbelt best development Macro-F1 remains {_pct(seatbelt['best_development_macro_f1'])}. Rights are unaccepted, all three subjects are repeatedly used development data, protected subjects were untouched, and no independent holdout exists.

`production_approved=false` and `final_status=RESEARCH_ONLY`. No active model/config changed, no DMD media was uploaded, no fine-tuning occurred, and no paper metric was presented as Roadwatch evidence. The next admissible step is rights-approved new subjects with human-validated labels, deduplication, frozen group splits, then a lightweight causal boundary/fusion experiment with full event IDs and one locked evaluation.
""",
    )

    source_paths = [
        "reports/v11_research/representation_experiments.json",
        "reports/v11_research/generalization_experiments.json",
        "reports/v11_research/fn_root_cause.json",
        "reports/v9_research/v9_diagnostics.json",
        "reports/v9_research/phone_context_event_replay.json",
        "reports/ai85_plus/runtime_benchmark.json",
    ]
    registry = {
        "schema_version": 1,
        "evidence_level": "LEVEL_2_DEVELOPMENT",
        "git_head_before_experiments": reps["git_head_before_experiment"],
        "dataset_hash": reps["dataset_hash"],
        "rights_status": dataset["rights_status"],
        "seed": reps["controlled_variables"]["seed"],
        "experiments": [
            {
                "id": "V11-A-E-REPRESENTATION",
                "hypothesis": reps["hypothesis"],
                "baseline": "V11-A_FROZEN_RESNET",
                "changed_factor": "representation only",
                "reproduction_command": "py -m training.dmd.v11_representation_experiments",
                "prediction_artifacts": [fold["prediction_path"] for row in reps["representations"] for fold in row["folds"]],
                "metrics": {row["representation_id"]: row["pooled"] for row in reps["representations"]},
                "decision": reps["decision"],
                "failure_analysis": "All initial candidates failed the worst-subject stability gate.",
            },
            {
                "id": "V11-E0-E4-GENERALIZATION",
                "hypothesis": gen["hypothesis"],
                "baseline": "E0_RAW_CAUSAL_MULTISCALE",
                "changed_factor": "one normalization or sampling strategy per row",
                "reproduction_command": "py -m training.dmd.v11_generalization_experiments",
                "prediction_artifacts": [fold["prediction_path"] for row in gen["representations"] for fold in row["folds"]],
                "metrics": {row["representation_id"]: row["pooled"] for row in gen["representations"]},
                "decision": gen["decision"],
                "failure_analysis": "E4 balancing regressed worst SAFE; E3 remains development-only.",
            },
            {
                "id": "V11-13FN-AUDIT",
                "hypothesis": "The 13 FNs span detection, association, action, fusion, and temporal stages.",
                "baseline": "Frozen 13-FN population",
                "changed_factor": "none; evidence consolidation",
                "reproduction_command": "py -m training.dmd.v11_fn_audit",
                "metrics": fn["summary"],
                "decision": "KEEP_DIAGNOSTIC_ONLY",
                "failure_analysis": fn["summary"]["failure_stage_counts_nonexclusive"],
            },
        ],
        "source_hashes": {path: sha256_file(ROOT / path) for path in source_paths},
        "fine_tuning_performed": False,
        "protected_holdouts_accessed": False,
        "production_approved": False,
        "final_status": "RESEARCH_ONLY",
    }
    _write_json(output / "experiment_registry.json", registry)
    return {
        "reports_written": 17,
        "selected_candidate": gen["selected_candidate"],
        "clip_macro_f1": selected["pooled"]["macro_f1"],
        "worst_subject_safe_recall": selected["worst_subject"]["safe_recall"],
        "production_approved": False,
        "final_status": "RESEARCH_ONLY",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps(run(args.output), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
