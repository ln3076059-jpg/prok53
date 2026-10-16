"""Render the requested evidence-backed reports from completed local runs."""

from __future__ import annotations

import json
import math
from pathlib import Path

from training.dmd.ai85_audit import artifact, read_json, write_json

ROOT = Path("reports/ai85_upgrade")


def percent(value: float) -> str:
    return f"{value * 100:.2f}%"


def table(headers: list[str], rows: list[list]) -> str:
    return "\n".join([
        "| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |",
        *("| " + " | ".join(str(cell).replace("\n", " ") for cell in row) + " |" for row in rows),
    ]) + "\n"


def dump(value) -> str:
    return "```json\n" + json.dumps(value, indent=2, ensure_ascii=False) + "\n```\n"


def write(name: str, title: str, body: str) -> None:
    (ROOT / name).write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def wilson(success: int, total: int) -> list[float]:
    z = 1.959963984540054
    p = success / total
    denominator = 1 + z * z / total
    center = (p + z * z / (2 * total)) / denominator
    half = z * math.sqrt(p * (1 - p) / total + z * z / (4 * total * total)) / denominator
    return [center - half, center + half]


def matched_truths(experiment: dict) -> set:
    return {(subject, m["ground_truth"]["start_seconds"], m["ground_truth"]["end_seconds"])
            for subject, row in experiment["per_subject"].items() for m in row["matches"]}


def main() -> None:
    results = read_json(ROOT / "metrics.json")
    native_path = ROOT / "native112/metrics.json"
    if native_path.is_file():
        native = read_json(native_path)
        results["variants"].extend(native["variants"][1:])
        results["comparison"].extend(native["comparison"][3:])
    environment = read_json(ROOT / "environment_audit.json")
    dataset = read_json(ROOT / "dataset_audit.json")
    errors = read_json(ROOT / "error_analysis.json")
    seatbelt = read_json(ROOT / "seatbelt_metrics.json")
    probe = read_json(ROOT / "phone_detector_probe.json")
    tests = read_json(ROOT / "test_results.json")
    benchmark_path = ROOT / "runtime_benchmark.json"
    runtime = read_json(benchmark_path) if benchmark_path.is_file() else None
    baseline = results["variants"][0]["experiments"][1]
    baseline_matches = matched_truths(baseline)
    deltas = []
    for variant in results["variants"]:
        for experiment in variant["experiments"]:
            matches = matched_truths(experiment)
            metric = experiment["pooled_metrics"]
            deltas.append({
                "experiment_id": experiment["experiment_id"],
                "recovered_ground_truth_events": sorted(matches - baseline_matches),
                "lost_ground_truth_events": sorted(baseline_matches - matches),
                "net_fp_change": metric["false_positives"] -
                                 baseline["pooled_metrics"]["false_positives"],
                "precision_wilson95": wilson(metric["true_positives"],
                                             metric["true_positives"] + metric["false_positives"]),
                "recall_wilson95": wilson(metric["true_positives"],
                                          metric["true_positives"] + metric["false_negatives"]),
            })
    write_json(ROOT / "event_deltas_and_intervals.json", {
        "rows": deltas, "interval_limitation":
        "Nominal Wilson intervals assume independent events; three clustered, repeatedly used "
        "subjects make them optimistic descriptions, not evidence of generalization.",
    })
    write_json(ROOT / "combined_metrics.json", results)
    comparison = table(
        ["Experiment", "TP", "FP", "FN", "Precision", "Recall", "F1", "FA/min", "SAFE", "Macro-F1"],
        [[r["experiment_id"], r["true_positives"], r["false_positives"], r["false_negatives"],
          percent(r["precision"]), percent(r["recall"]), percent(r["f1"]),
          r["false_alarms_per_minute"], percent(r["safe_driving_recall"]),
          percent(r["action_macro_f1"])] for r in results["comparison"]],
    )
    fn = [c for c in errors["cases"] if c["result"] == "FN"]
    write("01_GITHUB_AND_LOCAL_AUDIT.md", "GitHub and local audit", (
        f"Verified local branch `{environment['branch']}`, HEAD `{environment['git_head']}`. "
        "Remote candidate branch agrees with local HEAD; main is older. V7 is the latest "
        "available implementation; no V8 artifact was found.\n\n" + dump({
            k: environment[k] for k in ("remote_refs", "status_counts", "modified_source_paths",
                                        "python", "torch", "cuda_available", "memory", "disk")
        }) + "\nExisting edits and deletions were preserved. No reset, clean, checkout, "
        "restore, commit, push, production activation or new dataset download was performed. "
        "The audit records the working tree at audit time, including the new untracked work. "
        "Active weight hashes are in `environment_audit.json`."
    ))
    write("02_LATEST_BASELINE.md", "Latest verified baseline", (
        "V7 did not select an eligible replacement. Its comparator reuses the V6 hierarchical "
        "outer-LOSO predictions. Re-evaluation from original DMD annotations and stored event "
        "predictions exactly reproduces the stored metrics.\n\n" +
        dump(errors["recomputed_metrics"]) +
        "\nThere are 29 annotated phone events over 22.063 minutes and three development subjects. "
        "The frozen matcher counts repeated matches to one truth as duplicate false positives. "
        "`error_analysis.json` reconciles every match and false positive. SAFE recall in this "
        "comparison is the three-class coarse-head recall (12.57%); the six-class flat argmax "
        "SAFE recall is 14.82%. These quantities must not be interchanged."
    ))
    write("03_DATASET_AUDIT.md", "Dataset and preprocessing audit", (
        table(["Subject", "Cache frames", "Views", "Label mismatches", "Float boundary diffs"],
              [[s, v["samples"], ", ".join(v["available_views"]),
                v["annotation_label_mismatches"], v["float64_boundary_replay_disagreements"]]
               for s, v in dataset["subjects"].items()]) +
        "\nRGB and embeddings align exactly. Cached labels reproduce against the original "
        "float32 timestamp labeling path. Widening timestamps to float64 creates 13 sub-frame "
        "boundary disagreements; these are numeric replay differences, not evidence of human "
        "label errors. No cached or original label was changed.\n\n"
        "No exact RGB duplicate was found across the three subjects. Near duplicates were not "
        "measured. Overlapping windows exist within each subject; outer LOSO separates whole "
        "subjects before windows are built. Camera identity and driver identity cannot be "
        "disentangled with this pool. gC-14 lacks FACE/HANDS files.\n\n"
        "The original action path shrinks ROIs to 64×64 then upsamples to 112×112. The new "
        "native112 extraction uses original BODY pixels with identical ROI bounds, frame IDs, "
        "labels, normalization and pretrained ResNet weights. Per-file hashes, class counts and "
        "checks are in `dataset_audit.json` and `native112/embeddings/*_extraction.json`. "
        "No new dataset acquisition was performed; original source/license restrictions remain."
    ))
    fn_table = table(["Case", "Behavior", "Seconds", "Max PHONE", "Action failure"], [[
        c["case_id"], c["truth"]["source_action"].split("/")[-1],
        f"{c['truth']['start_seconds']:.3f}–{c['truth']['end_seconds']:.3f}",
        c["evidence"]["phone_probability_max"], c["action_branch_failure"],
    ] for c in fn])
    write("04_FALSE_NEGATIVE_ANALYSIS.md", "All false negatives", (
        fn_table + "\n" + dump(errors["stage_coverage"]) +
        "\nThese are parallel branch coverage measurements, not a causal sequential funnel. "
        "One short gC-14 event has no centered action sample. Other cases fail probability "
        "activation, sustained activation or matching. The source pixels for every FN midpoint "
        "were exported to `fn_original_rois/`, with exact frame IDs and hashes in "
        "`fn_frame_manifest.json`. Per-sample class probabilities, detector confidences and pose "
        "vectors are in `error_analysis.json`. The legacy cache lacks phone boxes, association "
        "IDs, calibrated fused scores and internal temporal states; those stages cannot be "
        "attributed from this cache alone."
    ))
    fp_table = table(["Subject", "Kind", "Start", "End", "Sampled true classes"], [[
        c["subject_id"], c["kind"], c["event"]["start_seconds"], c["event"]["end_seconds"],
        json.dumps(c["evidence"]["ground_truth_class_counts"]),
    ] for c in errors["false_positive_cases"]])
    write("05_FALSE_POSITIVE_ANALYSIS.md", "All false positives including duplicates", (
        fp_table + "\nThe baseline has four unmatched predictions plus one duplicate match. "
        "The inherited V7 diagnostic described only the four unmatched predictions; this "
        "implementation audits both categories and asserts their total equals evaluator FP. "
        "This corrects error accounting without changing any event metric or matcher. "
        "SAFE/OTHER confusion and fragmentation remain measurable failure modes."
    ))
    class_rows = []
    for variant in results["variants"]:
        class_rows.append([variant["variant"]["id"],
                           percent(variant["coarse_metrics"]["per_class"]["SAFE_DRIVING"]["recall"]),
                           percent(variant["clip_metrics"]["macro_f1"])])
    write("06_SAFE_DRIVING_CLASSIFICATION.md", "SAFE_DRIVING classification", (
        table(["Variant", "Coarse SAFE recall", "Six-class macro-F1"], class_rows) +
        "\nBaseline coarse confusion (rows/columns: SAFE, PHONE, OTHER):\n\n" +
        dump(errors["coarse_metrics"]["confusion_matrix"]) +
        "\nOf 668 SAFE clips, 84 are classified SAFE, 196 PHONE and 388 OTHER. SAFE is not a "
        "rare class. Additional weighted-loss continuation and attention do not solve the "
        "failure. The large gap between near-perfect inherited training accuracy and outer "
        "performance supports a subject/domain generalization problem. This is an inference, "
        "not proof of the visual cause. All new confusion matrices and per-class supports "
        "are preserved in `combined_metrics.json`."
    ))
    folds = [f for v in results["variants"][1:] for f in v["folds"]]
    write("07_ACTION_MODEL_EXPERIMENTS.md", "Controlled action-model experiments", (
        "Reused the existing frozen ImageNet ResNet18 and V6 BiGRU hierarchy. Each continuation "
        "starts from the checkpoint that excluded the same outer subject. Training uses only "
        "the other two subjects; outer embeddings are loaded after optimization. Five epochs, "
        "batch 16, learning rate 0.0001, sqrt-inverse weighted CE, no replacement sampling. "
        "AdamW is new: weight continuation rather than exact optimizer resume.\n\n"
        "MEAN_FINETUNE controls additional optimization. ATTENTION_FINETUNE replaces only mean "
        "pooling with zero-initialized learned temporal attention; initial outputs match the "
        "inherited model. NATIVE112_FINETUNE changes spatial sampling only, with mean pooling "
        "and identical continuation settings.\n\n" + table(
            ["Variant", "Outer subject", "Seed", "Train samples", "Train seconds"],
            [[f["variant_id"], f["heldout_subject"], f["training"]["seed"],
              f["training"]["samples"], round(f["training"]["training_seconds"], 3)] for f in folds]
        ) + "\nFinal epoch was fixed before evaluation. No outer-label early stopping or threshold "
        "search was performed. Checkpoint/prediction/input/config/source hashes and complete "
        "training histories are retained with each fold. Attention failure is retained. "
        "See `references.md` for primary research sources and applicability limits."
    ))
    write("08_PHONE_DETECTOR_EXPERIMENTS.md", "Phone detector resolution probe", (
        "Ran the unchanged phone detector on original BODY driver ROIs at each of 13 FN "
        "midpoints. Paired inputs use imgsz 384 and 960, confidence floor 0.10 and NMS IoU 0.7. "
        "All coordinates, confidences, input hashes, timings and model hash are saved.\n\n" +
        dump(probe["proposal_coverage"]) +
        "\nThese are unverified proposals on selected diagnostic snapshots, not detection recall "
        "or recovered events. Resolution alone did not increase proposal coverage in this probe. "
        "The single inspected gB-9 calling ROI visibly contains a handheld phone despite low "
        "action confidence; this supports further domain-specific representation research, "
        "but is not a new box annotation. No phone fine-tuning used inferred labels."
    ))
    write("09_FUSION_TEMPORAL_EXPERIMENTS.md", "Fusion and temporal decisions", (
        "Compared inherited union fusion with a preregistered negative-evidence rule: veto an "
        "object event only when at least half of overlapping action samples favor SAFE over "
        "PHONE and SAFE probability is at least 0.5. Missing action evidence cannot veto. "
        "Action-only evidence remains review-only.\n\n"
        "All runs use threshold 0.80, release ratio 0.65, two consecutive activation samples "
        "and the inherited two-second merge gap. Neither matcher nor ground-truth event "
        "boundaries changed. The veto retains original event bounds; no event stretching "
        "was introduced. Per-event veto reasons are stored.\n\n" + table(
            ["Experiment", "Recovered GT", "Lost GT", "Net FP change"],
            [[r["experiment_id"], len(r["recovered_ground_truth_events"]),
              len(r["lost_ground_truth_events"]), r["net_fp_change"]] for r in deltas]
        ) + "\nThe low-resolution continuations add a duplicate FP and do not recover a new truth. "
        "A separate temporal-policy search was not justified on three repeatedly used subjects. "
        "Onset latency in metrics is the inherited subject-mean statistic. Centered BiGRU clips "
        "also need roughly 2.016 seconds of lookahead; these are offline results, not live latency."
    ))
    seatbelt_data = read_json(
        "dataset/derived/seatbelt_classifier_v2/CLASSIFIER_DATASET_REPORT.json"
    )
    write("10_SEATBELT_EXPERIMENTS.md", "Seatbelt metric recomputation and training boundary", (
        f"Recomputed {seatbelt['sample_count']} validation predictions from the original CSV, "
        f"SHA256 `{seatbelt['source_sha256']}`. Raw macro-F1 "
        f"{percent(seatbelt['raw']['macro_f1'])}, balanced accuracy "
        f"{percent(seatbelt['raw']['balanced_accuracy'])}.\n\n" +
        table(["Class", "Support", "Precision", "Recall", "F1"], [[
            name, row["support"], percent(row["precision"]), percent(row["recall"]),
            percent(row["f1"]),
        ] for name, row in seatbelt["raw"]["per_class"].items()]) +
        f"\nPost-policy false-unfastened rate: "
        f"{percent(seatbelt['false_unfastened']['rate_among_true_fastened'])}. "
        "Confusion matrices, abstention denominators and ECE are in `seatbelt_metrics.json`.\n\n"
        "The available crop dataset contains 3,660 training images but zero UNCERTAIN examples; "
        f"its report says human review `{seatbelt_data['human_review_status']}` and "
        f"ready_for_training `{seatbelt_data['ready_for_training']}`. New three-state "
        "fine-tuning is blocked by this label/provenance gap. The 25 validation UNCERTAIN "
        "samples were not reused as training data. No threshold was tuned against this CSV, "
        "and frozen test data was not accessed."
    ))
    write("11_METRICS_COMPARISON.md", "Measured metrics comparison", comparison + (
        "\nComplete per-subject metrics, confusion matrices, event predictions and matching "
        "evidence: `combined_metrics.json`. Recovered/lost truths, FP differences and nominal "
        "95% Wilson intervals: `event_deltas_and_intervals.json`. The intervals assume independent "
        "events and are optimistic for three subjects. Clip macro-F1 does not replace event F1."
    ))
    write("12_GENERALIZATION_VALIDATION.md", "Generalization boundary", (
        "All measured candidates are iterative development LOSO on gB-9, gZ-36 and gC-14. "
        "Their three outer folds contain no subject overlap with the corresponding inherited "
        "checkpoint or continuation training set. gZ-37 and gE-28 were not read for training, "
        "feature extraction, calibration, evaluation or model selection. Filesystem inventory "
        "alone does not consume a holdout.\n\n"
        "No approved new independent subject pool is available. The repeated reuse of these "
        "three subjects precludes an independent 85% claim, even if a development score were "
        "to exceed it. Next work needs licensed new subjects, human-approved annotations, "
        "subject splits fixed before training, and one untouched evaluation after model freeze."
    ))
    inference_rows = [[f["variant_id"], f["heldout_subject"],
                       f["inference"]["inference_seconds"], f["inference"]["clips_per_second"]]
                      for f in folds]
    write("13_PERFORMANCE_BENCHMARK.md", "Measured local performance", (
        table(["Variant", "Subject", "Head inference seconds", "Cached clips/s"], inference_rows) +
        "\nCached-head throughput excludes decoding and ResNet inference. Native112 extraction "
        "metadata records complete video-decoding-plus-embedding time per subject. CPU, PyTorch, "
        "memory and disk snapshots are in `environment_audit.json`. CUDA is unavailable.\n\n" +
        (dump(runtime["full_backend_pipeline"]) +
         "\nThis is a short real VideoAnalyzer + SQLite/evidence-path smoke benchmark. "
         "It measures the active V2 backend, not a deployed research action branch. "
         "No research checkpoint was substituted into the backend."
         if runtime else "No new full-backend benchmark artifact was produced. Historical "
         "V6 throughput is not presented as a new measurement.")
    ))
    test_summary = {key: {k: v for k, v in value.items() if k != "failures"}
                    for key, value in tests.items()}
    for value in test_summary.values():
        if "counts" in value:
            counts = value["counts"]
            counts["passed"] = counts["tests"] - sum(
                counts[k] for k in ("errors", "failures", "skipped")
            )
    classification = read_json(ROOT / "test_failure_classification.json")
    write("14_TEST_RESULTS.md", "Executed software checks", dump(test_summary) +
          dump(classification) + (
        "\nLogs and JUnit XML are retained. `test_results.json` contains all failure details. "
        "The full suite is reported with its actual status, including failures from deleted "
        "legacy `datasets/` paths. Tests were not skipped or rewritten to hide those failures. "
        "The targeted suite includes the new regression checks, V5–V7 compatibility, holdout "
        "protection, runtime configuration, backend queue/integration and experiment runners."
    ))
    best = max(results["comparison"], key=lambda r: r["f1"])
    write("15_FINAL_REPORT.md", "AI85 upgrade: actual results and remaining work", (
        "The requested audit, code changes, controlled CPU training, detector probe and "
        "evaluation were executed. The 85% target is not established. No candidate is "
        "production-approved.\n\n" + comparison +
        f"\nHighest observed development F1 in this run: `{best['experiment_id']}`, "
        f"Precision {percent(best['precision'])}, Recall {percent(best['recall'])}, "
        f"F1 {percent(best['f1'])}. This is descriptive and does not select a production model.\n\n"
        "Implemented checkpoint-compatible temporal attention, deterministic fine-tuning with "
        "explicit subject isolation, direct original-resolution ROI embedding extraction, "
        "negative-evidence fusion, and an error audit that includes duplicate false positives. "
        "All 13 FN cases have original ROI evidence. Failed experiments are retained.\n\n"
        "The remaining evidence gaps are new independent subjects, representative approved "
        "UNCERTAIN seatbelt training labels, verified phone boxes/association evidence, "
        "and stronger cross-subject representations. Repeating threshold searches on these "
        "three subjects would not resolve those gaps. See `14_TEST_RESULTS.md` for actual "
        "test outcomes and unresolved inherited failures.\n\n"
        "Reproduce with Python 3.11 from the repository root; output directories must be fresh "
        "because existing fold checkpoints are protected from overwrite:\n\n"
        "```powershell\n"
        "py -3.11 -m training.dmd.run_ai85_upgrade --stage audit\n"
        "py -3.11 -m training.dmd.run_ai85_upgrade --stage train\n"
        "py -3.11 -m training.dmd.ai85_detector_probe\n"
        "py -3.11 -m training.dmd.ai85_native_embeddings --subject gB-9\n"
        "py -3.11 -m training.dmd.ai85_native_embeddings --subject gZ-36\n"
        "py -3.11 -m training.dmd.ai85_native_embeddings --subject gC-14\n"
        "py -3.11 -m training.dmd.run_ai85_upgrade --stage train "
        "--config models/model_config_ai85_native112.yaml --output reports/ai85_upgrade/native112\n"
        "py -3.11 -m training.evaluate_seatbelt_v6 "
        "--output reports/ai85_upgrade/seatbelt_metrics.json\n"
        "py -3.11 -m training.benchmark_v6_runtime --frames 12 "
        "--output reports/ai85_upgrade/runtime_benchmark.json\n"
        "py -3.11 -m training.dmd.verify_ai85\n"
        "py -3.11 -m training.dmd.render_ai85_reports\n"
        "```\n"
    ))
    review = Path("reports/ai85")
    review.mkdir(exist_ok=True, parents=True)
    (review / "01_ERROR_ANALYSIS.md").write_text(
        "# AI85 event error analysis\n\n" + fn_table + "\n" + fp_table +
        "\nDetailed traces and stage coverage: [error_analysis.json]"
        "(../ai85_upgrade/error_analysis.json). Original FN ROIs: "
        "[manifest](../ai85_upgrade/fn_frame_manifest.json).\n", encoding="utf-8",
    )
    assets = [artifact(p) for p in sorted(ROOT.rglob("*")) if p.is_file() and
              p.name != "artifact_manifest.json"]
    write_json(ROOT / "artifact_manifest.json", {"artifacts": assets})
    print(f"Rendered 15 reports; inventoried {len(assets)} artifacts.")


if __name__ == "__main__":
    main()
