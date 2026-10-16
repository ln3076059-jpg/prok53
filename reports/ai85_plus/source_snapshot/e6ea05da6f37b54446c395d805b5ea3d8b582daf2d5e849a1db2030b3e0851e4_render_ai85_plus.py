"""Render the sixteen required reports from measured JSON artifacts."""
from __future__ import annotations

from pathlib import Path

from training.dmd.ai85_audit import read_json, write_json

OUT = Path("reports/ai85_plus")


def table(headers, rows):
    return "\n".join(["| " + " | ".join(headers) + " |",
                      "| " + " | ".join(["---"] * len(headers)) + " |",
                      *("| " + " | ".join(str(c).replace("|", "/") for c in row) + " |" for row in rows)])


def pct(value):
    return f"{100 * value:.2f}%"


def report(name, title, body):
    (OUT / name).write_text(f"# {title}\n\n{body.strip()}\n", encoding="utf-8")


def run():
    audit = read_json(OUT / "audit.json")
    models = read_json(OUT / "checkpoint_metadata.json")["models"]
    temporal = read_json(OUT / "temporal/temporal_results.json")
    visual = read_json(OUT / "phone_spatial_probes.json")
    seatbelt = read_json(OUT / "seatbelt_results.json")
    diagnostic = read_json(OUT / "evidence_diagnostics.json")
    checks = read_json(OUT / "test_results.json")
    runtime = read_json(OUT / "runtime_benchmark.json")
    kaggle = read_json(OUT / "kaggle_job.json") if (OUT / "kaggle_job.json").exists() else None
    kaggle_note = "No remote GPU result was available when rendering this report."
    if kaggle:
        kaggle_note = (
            f"Private Kaggle job [{kaggle['kernel']}]({kaggle['url']}): "
            f"{kaggle['status']}. It runs frozen phone inference on 64 legacy validation "
            "images already in the account. No DMD media, holdouts or credentials were uploaded."
        )
        if kaggle.get("result"):
            gpu = read_json(kaggle["result"]["path"])
            kaggle_note += "\n\n" + table(
                ["GPU", "Variant", "Images", "Detector-call FPS", "p50 ms", "p95 ms"],
                [[gpu["gpu"], v["id"], v["images"], round(v["detector_call_fps"], 2),
                  round(v["p50_ms"], 2), round(v["p95_ms"], 2)] for v in gpu["variants"]]
            )
            kaggle_note += (
                "\n\nGPU and local CPU probes use different image sets; their throughput is "
                "not a controlled hardware speedup comparison. GPU detector-call FPS is not "
                "full-system FPS, and no AP/accuracy target was measured on Kaggle."
            )
    previous = read_json("reports/ai85_next/analysis.json")
    variants = temporal["variants"]
    baseline = variants[0]["pooled"]
    binary = previous["event_variants"]["SN_PHONE_INHERITED_POLICY_FUSION"]["pooled_metrics"]
    exploratory = previous["event_variants"]["SN_PHONE_EXPLORATORY_POLICY"]["pooled_metrics"]
    assert [baseline[k] for k in ("true_positives", "false_positives", "false_negatives")] == [16, 5, 13]
    assert [binary[k] for k in ("true_positives", "false_positives", "false_negatives")] == [16, 4, 13]
    metrics = []
    for name, m, safe, source in [
        ("Historical V6 / T0", baseline, variants[0]["coarse_safe_recall"], "replayed frozen comparator"),
        ("AI85 next controlled binary", binary, None, "historical consumed development; binary SAFE not six-class SAFE"),
        ("AI85 next exploratory", exploratory, None, "INELIGIBLE posthoc policy exploration"),
    ] + [(v["id"], v["pooled"], v["coarse_safe_recall"], "consumed development; no selection") for v in variants[1:]]:
        metrics.append({"candidate": name, **m, "coarse_safe_recall": safe, "source": source})
    write_json(OUT / "metrics_comparison.json", {"event_metrics": metrics,
               "direct_coarse_safe_recall_separate_experiment": previous["direct_coarse_metrics"]["per_class"]["SAFE_DRIVING"]["recall"],
               "seatbelt": seatbelt["variants"], "independent_validation": "INDEPENDENT_AI85_NOT_ESTABLISHED"})
    comparison = table(["Candidate", "TP", "FP", "FN", "P", "R", "F1", "FA/min", "Coarse SAFE R"], [
        [m["candidate"], m["true_positives"], m["false_positives"], m["false_negatives"],
         pct(m["precision"]), pct(m["recall"]), pct(m["f1"]), m["false_alarms_per_minute"],
         pct(m["coarse_safe_recall"]) if m["coarse_safe_recall"] is not None else "not comparable"] for m in metrics])
    common = ("Status: `INDEPENDENT_AI85_NOT_ESTABLISHED`. All new AI measurements use already-consumed "
              "development data. No production model was approved. Machine-readable evidence is in this directory.")
    report("01_PROJECT_AUDIT.md", "Project and resource audit", f"""
Source commit: `{audit['git_head']}`. Local and remote branch matched before changes.
Branch: `roadwatch-v4-candidate`. The user's modified prompt was retained.

The read-only audit found {audit['model_count']} checkpoint files and {audit['data_trees']['dataset']['files_excluding_protected_paths']} files under `dataset/`, excluding protected paths. `datasets/` is absent.
RAM: {audit['ram_total_mb']:.1f} MiB. Audit free disk: {audit['disk_free_gb']:.2f} GiB. Runtime hardware CUDA available: {runtime['hardware']['cuda_available']}. No external training mount was configured.

Implemented: explicit data-root resolution, T0–T5 visual sampling experiments, evidence-bounded confirmation, ROI-to-frame box mapping, missing-view abstention, frozen detector/seatbelt inference and artifact verification.
Production integration remains gated because no new candidate is eligible. Existing model loading and backend workflow were exercised by system smoke and full pytest.

[Audit JSON](audit.json), [test evidence](15_TEST_RESULTS.md), [registry](../../models/model_registry_ai85_plus.json).

{common}
""")
    model_table = table(["Path", "Architecture", "Classes", "Input", "Epochs", "SHA256"], [
        [m["path"], m["architecture"], m["class_mapping"], m["input_resolution"], m["epochs"], m["sha256"]] for m in models])
    report("02_MODEL_WEIGHT_INVENTORY.md", "Actual checkpoint inventory", f"""
{len(models)} actual files were hashed and inspected; duplicates retain their separate paths. Full checkpoint fields, training data references, embedded validation metrics, optimizer presence and approval status are in [checkpoint_metadata.json](checkpoint_metadata.json).
Null fields mean unavailable checkpoint metadata. Generic COCO/notebook weights are not assumed to belong to Roadwatch. Embedded training AP is historical and is not a new lawful-GT evaluation.
Active phone weights are single-class `phone`, seatbelt detector is `occupant_upper_body`, and the classifier has three belt states. Occupant AP is not belt-state AP.
No exact optimizer/scheduler resume was verified. Frozen weights were reused; no training was run.

{model_table}
""")
    migration = diagnostic["migration_summary"]
    report("03_DATASET_VALIDATION.md", "Dataset roots, labels and migration", f"""
The local root is `dataset/`; `datasets/` does not exist. `ROADWATCH_DATA_ROOT` can explicitly select another root; ambiguous conventional roots fail with an actionable error. Explicit caller paths are preserved.

Historical deleted-path audit: {migration}. The detailed old-Git-blob and local SHA256 comparison is in [evidence_diagnostics.json](evidence_diagnostics.json). No dataset was restored, moved, deleted or relabeled. A byte mismatch must be investigated individually; existence alone is not checksum equivalence.
The follow-up [migration difference audit](dataset_migration_differences.json) confirms all 67 byte mismatches differ only in line endings or outer whitespace. All 65 JSON/JSONL mismatches are semantically equal; the other two are CSV files with newline-only differences.

Policies, schemas, manifests, stale path references and hashes are listed in [audit.json](audit.json). This stage fixes the schema/policy consumers and readiness audit exercised by tests; historical manifests and obsolete training scripts can still contain old `datasets/` references.
DMD remains `RIGHTS_ACCEPTED=false`, `CANONICAL_ELIGIBLE=false`, `TEMPORAL_DEVELOPMENT_ELIGIBLE=true` in the intake record. There are three permitted consumed development subjects, seven BODY/FACE/HANDS videos, and 29 phone events in the evaluated protocol. gC-14 has BODY only. Protected directories were pruned before content walking.

The older seatbelt classifier dataset has 0 train UNCERTAIN crops and `ready_for_training=false`. The actual pending-approval pretrain bundle has 100 train, 25 val and 25 test UNCERTAIN crops, and its labels are explicitly model-assisted pending approval. It has no approved independent subject split.
Phone bootstrap reports 70,282 near-duplicate pairs and grouped splits; these historical clustering facts do not establish unseen subject/camera generalization. No new near-duplicate certification or new subject consent was fabricated.
""")
    spatial = table(["Variant", "Frames", "Frames with proposals", "Proposals", "p50 ms", "p95 ms"], [
        [v["id"], v["frames"], v["frames_with_proposal"], v["proposals"],
         round(v["latency"]["p50_ms"], 2), round(v["latency"]["p95_ms"], 2)] for v in visual["summary"]])
    report("04_ZERO_RETRAINING_ABLATIONS.md", "Measured zero-retraining probes", f"""
All weights remained frozen. Ultralytics {visual['ultralytics']}; prediction confidence 0.10, class-aware NMS IoU 0.70 unless the explicit agnostic variant is used. No AP curve is calculated from this confidence cutoff.
The checkpoint training metadata says imgsz=960. The measured `A2_full_960` row is its original-resolution reference; `A0_full_640` is the matched 640 reference for the TTA/NMS probes. These names are retained in raw artifacts to avoid changing measured history. TTA was compared only at 640, not asserted at 960.
Thirteen original-video midpoints were selected for FN diagnosis, not for detector validation or model selection. All timestamps are checked inside the corresponding interval.

{spatial}

No A5 combination was selected because no legitimate validation gain was established. No gain in phone-event recall can be inferred from proposal counts. [Raw boxes and timings](phone_spatial_probes.json).
""")
    fn_rows = []
    for case in diagnostic["false_negative_cases"]:
        t0, t1 = case["sampling"]["T0"], case["sampling"]["T1"]
        fn_rows.append([case["case_id"], f"{case['truth']['end_seconds'] - case['truth']['start_seconds']:.3f}",
                        t0["centered_samples_in_interval"], t1["centered_samples_in_interval"],
                        t1["raw_phone_probability_max"], t1["matched"], case["visual_review"]])
    report("05_PHONE_FALSE_NEGATIVE_ANALYSIS.md", "Thirteen original false negatives", f"""
Original-video midpoint evidence and dense temporal traces are linked in the machine artifacts. Visual comments are inherited frame inspections, not human confirmation of every frame or revised labels. Lighting/angle, true phone pixels and hand/face association lack validated annotations and remain unknown; proposal sizes are reported separately.

{table(['Case', 'Duration s', 'T0 centers', 'T1 centers', 'T1 max raw phone p', 'T1 matched', 'Midpoint observation'], fn_rows)}

The 0.403-second gC-14 interval gains dense centers but remains unmatched. Across all four subsecond events, T1 achieves coverage 4/4 and matching 0/4. Temporal coverage is therefore only one bottleneck; raw action evidence, smoothing, minimum support and frozen matcher compatibility still limit detection.
ROI clipping, blur and occlusion are visible in existing examples. Padding was measured and did not improve proposal coverage over the unpadded driver crop in this midpoint diagnostic. No geometry hard gate was added.

[Per-case traces](evidence_diagnostics.json), [original prior ROI manifest](../ai85_upgrade/fn_frame_manifest.json), [new detector boxes](phone_spatial_probes.json).
""")
    duration_table = table(["Variant", "<1s matched/total", "1–3s", "3–10s", ">10s", "Recovered", "Lost", "Duplicate matches"], [
        [v["id"], *(f"{v['duration_buckets'][b]['matched']}/{v['duration_buckets'][b]['total']}" for b in ("<1", "1-3", "3-10", ">10")),
         len(v["recovered"]), len(v["lost"]), v["duplicates"]] for v in variants])
    report("06_ADAPTIVE_TEMPORAL_SAMPLING.md", "T0–T5 controlled temporal sampling", f"""
T0: original 15-frame stride. T1: 3-frame stride. T2: baseline plus current motion triggers. T3: baseline plus 6-frame stride. T4: motion-triggered 0.4s sampling burst. T5: T4 with two-sample causal confirmation, no confirmation-stage future smoothing or endpoint extrapolation. Inherited object/action late fusion remains common to all variants.
Motion is the normalized mean absolute grayscale difference on the 3-frame observation grid. Threshold 0.035 and burst length were engineering constants frozen before the experiment; they were not searched by GT labels.
The sampler is prefix-causal and has no annotation input. The inherited centered BiGRU is not causal: each action center uses ±60 original frames, i.e. 2.016s of future context. T0–T4 additionally inherit one-sample centered smoothing (about 0.504s at T0). Matcher onset latency is not usable online alarm delay; context availability must also be counted. T5 confirmation timestamps are retained where present, while object-fusion emission latency is not measured.
Original 15-frame embeddings are reused byte-for-byte. T0 probability discrepancy is at most {max(s['T0_max_probability_error'] for s in temporal['subjects'].values()):.9g}, and its TP/FP/FN reproduce 16/5/13.

{duration_table}

{comparison}

T1/T3 recover one long event and lose zero baseline matches; subsecond recall remains 0%. T1 adds eight net FP, so its F1 is lower. Exact new/removed FP intervals, per-subject scores, masks and full match evidence are in [temporal_results.json](temporal/temporal_results.json).
All contexts were extracted in the benchmark. Selected centers alone cannot be presented as actual encoder savings or full-system FPS.
""")
    report("07_OBJECT_DETECTION_EXPERIMENTS.md", "Detector validation limits", f"""
New governed detector AP50/AP50–95 and per-class P/R: **BLOCKED — no approved bbox validation ground truth established**. DMD action intervals are not phone boxes. The phone bootstrap explicitly restricts its proposals from being reported as final GT. Pending-approval detector labels cannot certify the requested lawful-GT 85% targets.
Embedded checkpoint metrics are preserved in checkpoint_metadata.json as historical training metadata only. A phone AP50 entry in that metadata cannot establish event F1 or roadside performance. The seatbelt detector localizes upper bodies; it is not a belt-state detector.
The installed 8.3.187 validator sets unspecified non-OBB validation confidence to 0.001 (OBB: 0.01), confirmed from its installed source in validation_defaults.json. conf=0.25 was not used as an AP default. NMS IoU=0.70 in spatial probes is distinct from AP IoU=0.50 or the 0.50–0.95 sweep.
Measured spatial diagnostics: seven configurations × thirteen FN midpoints = 91 predictions, with original-frame boxes reconstructed and clamped after crop inference. Static driver ROI containment is not verified occupant-phone association. Torso/hand/face localization improvements require validated localizations and were not claimed from static crops.

{spatial}
""")
    report("08_ACTION_CLASSIFICATION_EXPERIMENTS.md", "Action representation and confidence diagnostics", f"""
The hierarchical ResNet18/BiGRU checkpoints were reused, with the same six-class map and physical temporal context. Spatial input remains 64px downsample then bilinear 112px: this experiment isolates sampling. Mean/attention/native112 training failures were not repeated.
The earlier subject-normalized direct coarse probe has SAFE recall {pct(previous['direct_coarse_metrics']['per_class']['SAFE_DRIVING']['recall'])}, separately from the binary branch event F1 {pct(binary['f1'])}. Improving SAFE argmax does not ensure sustained PHONE probability crosses the inherited 0.80 event threshold. Coarse PHONE sums four fine probabilities; fine SAFE argmax is a different decision.
Prior representation diagnostics: {previous['subject_bias']}. Whole-video normalization uses future unlabeled statistics and remains an offline transductive experiment.

{table(['Variant', 'Six-class Macro-F1', 'Clip accuracy', 'Coarse SAFE recall', 'Evaluated centers'], [[v['id'], pct(v['classification']['macro_f1']), pct(v['classification']['clip_accuracy']), pct(v['coarse_safe_recall']), v['classification']['evaluated_clips']] for v in variants])}

Different samplers evaluate different center distributions; these clip metrics are descriptive and not a common-grid classifier ranking. Full per-class confusion matrices remain in temporal_results.json. No new classifier was trained without approved labels and a legitimate independent validation split. Behavior ROI and temporal dynamics informed implemented probes, not paper reproduction. [Primary sources](references.md).
""")
    report("09_MULTI_VIEW_FUSION.md", "Availability-aware multiview evidence", f"""
BODY/FACE/HANDS availability is read from the actual legacy records. Missing views are represented by `None` and excluded from confidence fusion denominators. gC-14 has BODY only; a missing FACE/HANDS camera never contributes negative evidence.

{table(['Subject', 'BODY available', 'FACE available', 'HANDS available'], [[s, *(v['availability'][k]['available_records'] for k in ('body','face','hands'))] for s,v in diagnostic['multiview'].items()])}

The new availability-aware helper was exercised on cached evidence; its aggregate maximums are diagnostic confidence summaries, not calibrated probabilities or an integrated event gain. Existing cache lacks aligned phone boxes plus hand/face landmarks and occupant tracks. No normalized ear/phone distances, head pose, MediaPipe integration or reliability calibration was fabricated. No geometry threshold was selected on protected subjects, and occluded phones were not hard-rejected by geometry.
Further multi-task fusion requires quality-checked synchronized views and landmarks. Sources in references.md motivate this direction only.
""")
    report("10_FINE_TUNING_RESULTS.md", "Selective fine-tuning decision", """
No training was run in V8. F0 reuses the inspected original checkpoints. F1–F5 are BLOCKED pending confirmed labels/license, a subject-disjoint training/validation split and resource approval for the resulting dataset. CPU was sufficient for frozen inference; CUDA is unavailable locally.
Existing pending-approval labels do not satisfy the prompt's confirmed-GT prerequisites. No new legitimate subject pool was found. Repeating failed mean/attention/native112 training or searching thresholds on the same subjects would not establish generalization.
No active or historical checkpoint was overwritten. `checkpoint_metadata.json` records optimizer/scheduler presence. Any later use of best.pt is a warm start unless complete training state is independently verified; this stage establishes no exact resume.
GPU ablations should freeze split and lineage first, then vary original resolution, higher resolution, reviewed hard examples and augmentation separately. No gain or training time is promised. `model_config_ai85_plus.yaml` keeps fine-tuning disabled and records prerequisites.
""")
    belt_rows = []
    for variant in seatbelt["variants"]:
        ev = variant["evaluation"]
        belt_rows.append([variant["id"], pct(ev["raw"]["macro_f1"]), pct(ev["raw"]["balanced_accuracy"]),
                          *(pct(ev["raw"]["per_class"][k]["recall"]) for k in ("seatbelt_fastened", "seatbelt_unfastened", "uncertain_or_occluded")),
                          pct(ev["false_unfastened"]["rate_among_true_fastened"]), pct(ev["abstention"]["output_uncertain_rate"])])
    report("11_SEATBELT_RESULTS.md", "Seatbelt inference and uncertainty", f"""
Fresh inference on the same 641 consumed validation crops: original 224px, 320px, and average of original/horizontal flip at 224px. Each variant has a prediction CSV, per-class P/R/F1, raw and policy confusion matrices, ECE, false-unfastened and abstention in [seatbelt_results.json](seatbelt_results.json).
Labels are model-assisted pending approval. They are not independent confirmed-GT performance. Original CSV replay is included separately because current inference reproduction can differ from stored probabilities.

{table(['Variant', 'Raw Macro-F1', 'Raw balanced acc', 'Raw FASTENED R', 'Raw UNFASTENED R', 'Raw UNCERTAIN R', 'Policy false-unfastened', 'Policy abstention'], belt_rows)}

False-unfastened and abstention columns use the inherited 0.50/0.60 operating policy; raw recall/Macro-F1 columns use argmax. Both complete metric sets are in JSON. No thresholds were tuned here. Horizontal flips are exploratory for asymmetric cabin imagery and are not selected for production.
Images are existing torso crops; original-frame localization quality and temporal smoothing cannot be established from unordered crop validation. Occlusion/dark clothing provenance is incomplete. Invisible belt remains uncertain, never automatic unfastened. Additional attention/classifier training remains blocked by confirmed-label and independence prerequisites.
""")
    report("12_METRICS_COMPARISON.md", "Comparable event metrics", f"""
{comparison}

Historical direct coarse SAFE recall 70.81% is a separate experiment and is not attached to the binary event row. Six-class action and seatbelt metrics remain separate from event metrics. The exploratory 88.89% precision row is ineligible for selection.
Per-subject event evidence and recovered/lost event identities are in temporal/temporal_results.json. T1/T3 recover one long event without losing baseline TP, but add FP and decrease F1. No new eligible candidate is selected. [Machine table](metrics_comparison.json).
""")
    report("13_GENERALIZATION_VALIDATION.md", "Generalization and split protection", f"""
{common}

No new independently licensed subject holdout was established. gB-9, gZ-36 and gC-14 are iterative development subjects. LOSO checkpoint inference does not make repeatedly examined subjects an untouched final test set. All V8 candidate metrics are descriptive development ablations.
gZ-37 and gE-28 are permanently protected. New extraction checks subject identity before filesystem access; tests verify both rejected subjects cause no file access. Inventory prunes protected directories and only lists immediate subject directory names.
No protected-subject training, extraction, tuning, threshold selection or review was performed. No data was downloaded from linkdmd.md or unverified links. A future eligible dataset must establish rights/consent, real identity/camera grouping, duplicate checks and locked splits before training, followed by one untouched evaluation after model freeze.
""")
    pipeline = runtime["full_backend_pipeline"]
    report("14_RUNTIME_BENCHMARK.md", "Measured runtime and scope", f"""
Actual current development backend VideoAnalyzer pipeline: {pipeline['frames']} frames, {pipeline['processing_seconds']} seconds, **{pipeline['processing_fps']} processing FPS**, with SQLite and evidence path. Detection rows: {pipeline['database_detection_rows']}; event rows: {pipeline['database_event_rows']}. This short cabin clip is a smoke benchmark and is not roadside accuracy evidence.
Detector/context p50/p95, decode, encoder, cached temporal head, seatbelt direct probes, fusion direct probes, evidence writes, RAM and CUDA/VRAM are in [runtime_benchmark.json](runtime_benchmark.json). Zero in-pipeline calls mean that component was unexercised; separately timed direct probes do not establish loaded event-path throughput.
Adaptive research extraction:

{table(['Subject', 'Dense samples', 'Extraction seconds', 'Peak RSS MiB'], [[s, v['extraction']['dense_samples'], round(v['extraction']['seconds'], 3), round(v['extraction']['peak_rss_mb'], 1)] for s,v in temporal['subjects'].items()])}

Dense extraction uses batched frozen features and shared contexts. It is not full-system FPS. T1 uses about 5× the action centers; unique required context embeddings are recorded separately per subject and sampler. A real scheduler with context reuse would be needed to realize savings. No integrated adaptive full-pipeline speed claim is made because no eligible adaptive model was promoted.
Spatial TTA/resolution/ROI timings are measured in phone_spatial_probes.json; no universal speed/quality gain is asserted. CPU benchmarks include local process load and have small-sample limits.

{kaggle_note}
""")
    report("15_TEST_RESULTS.md", "Verification results", f"""
{table(['Check', 'Result', 'Evidence'], [[k, v.get('status', v.get('summary','unknown')), v.get('evidence', v.get('command',''))] for k,v in checks.items()])}

Full pytest counts are parsed from JUnit XML rather than terminal progress. Earlier failed attempts are retained; final counts refer to the final run. The prior 37 failures / 74 errors from missing dataset paths were resolved by root-aware schema/policy consumers. The readiness test previously skipped due to its stale path is now executed.
Backend smoke uses temporary SQLite/uploads/evidence and checks real startup, liveness, readiness, OpenAPI and authentication rejection. Full pytest covers queue/review/evidence and split guards. Frontend TypeScript/Vite build completed successfully. Browser-driven E2E remains BLOCKED because no browser E2E harness is configured; API/inference smoke is reported separately.
No test was deleted or weakened to improve PASS counts. Ground truth and event matcher remain unchanged. [Machine test results](test_results.json).
""")
    report("16_FINAL_RESEARCH_REPORT.md", "Roadwatch AI85+ V8 final research report", f"""
**Status: `INDEPENDENT_AI85_NOT_ESTABLISHED`. No new production-approved or selected candidate.**

Delivered working research code and measurements: safe root resolution, all-checkpoint inventory, original-video dense/adaptive sampling, bounded confirmation, ROI box restoration, missing-view abstention, 91 frozen detector spatial probes, three fresh seatbelt inference runs, original FN traces, system smoke and runtime benchmark.

{comparison}

The temporal hypothesis is partly falsified: dense sampling covers all four subsecond events but recognizes none under the frozen model/protocol. It recovers one long event while adding enough false alarms to reduce F1. Representation confidence and phone localization remain stronger bottlenecks than sample coverage alone. Simple motion triggers and confirmation do not improve event F1 here.
The controlled historical binary branch remains a useful development comparator (80% P, 55.17% R, 65.31% F1), not a new approved model. Direct coarse SAFE 70.81% is a separate result. Seatbelt metrics are in report 11 and cannot certify the 85% target with pending-approval labels.

Targets were not established independently. Detector governed AP is blocked by approved bbox GT; new training is blocked by confirmed labels and independent splits; independent final evaluation is blocked by a new lawful holdout. Existing protected subjects were not reused. The source/test path defects were resolved, while historical manifests retain their original lineage and paths.
Next admissible work is reviewed phone boxes/hard negatives and complete three-class seatbelt labels with real subject/camera grouping, then locked validation for representation and ROI experiments. No further threshold search on these three subjects was performed.

Evidence: reports 01–15, checkpoint_metadata.json, metrics_comparison.json, temporal predictions/masks/matches, seatbelt prediction CSVs, source/model/input hashes, artifact validation and [reproduction commands](reproduce.ps1). [Primary research references](references.md).

{kaggle_note}
""")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    OUT = parser.parse_args().output
    run()
