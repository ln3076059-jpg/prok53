"""Render Roadwatch V9 scientific reports from measured artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT = ROOT / "reports" / "v9_research"


def _read(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def _pct(value: float | None) -> str:
    return "—" if value is None else f"{100 * value:.2f}%"


def _table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    lines.extend("| " + " | ".join(str(cell) for cell in row) + " |" for row in rows)
    return "\n".join(lines)


def render(output: Path = DEFAULT_OUTPUT) -> list[Path]:
    diagnostics = _read(output / "v9_diagnostics.json")
    fn = diagnostics["phone_false_negatives"]
    seatbelt = diagnostics["seatbelt"]
    representation = diagnostics["representation"]
    fusion = diagnostics["fusion_ablation"]
    gpu = diagnostics["runtime"]["kaggle_gpu"]
    pose_available = sum(item["pose"]["status"] == "AVAILABLE" for item in fn["cases"])
    geometry_blocked = sum(
        item["proposal_geometry"]["status"] == "BLOCKED_NO_PHONE_BOX" for item in fn["cases"]
    )
    geometry_unverified = sum(
        item["proposal_geometry"]["status"] == "UNVERIFIED_NO_OCCUPANT_TRACK"
        for item in fn["cases"]
    )
    handheld_context = sum(
        item["proposal_geometry"].get("geometry_context") == "HANDHELD_USE" for item in fn["cases"]
    )

    literature_rows = [
        [
            "DualScope (AAAI 2026)",
            "7 DDAR datasets; video action classification",
            "Behavior-centric SAM distillation + crops; motion/difference encoder + temporal-pyramid Mamba",
            "Paper classification metrics; not Roadwatch event F1",
            "Paper + HF checkpoint card; linked GitHub returned 404 during audit",
            "Test crop/motion/multi-scale hypotheses under Roadwatch protocol",
        ],
        [
            "Multi-Task Fusion (Sensors 2025)",
            "DMD; AUC_V2/SAMDD robustness",
            "3D ResNet-18 + ConvLSTM/BiLSTM on BODY/FACE/HANDS; action/gaze/hand-wheel fusion",
            "Paper attentiveness F1 98.28% at threshold 0.5; different task",
            "Open article; data on request; no official code/weights located",
            "Reliability-aware available-view fusion; gC-14 has BODY only",
        ],
        [
            "DriveCLIP (IEEE T-ITS 2024)",
            "DMD, StateFarm, SynDD1, SAM-DD",
            "CLIP visual-language frame classifier; separate VideoCLIP reference",
            "Frame/video classification; not event F1",
            "MIT repo; five small .pkl artifacts; no frozen CLIP artifact installed locally",
            "Controlled feature baseline after artifact/hash registration",
        ],
        [
            "DGMB-Net (Sensors 2024)",
            "Bayonet-Drivers 10k images/3k labeled; State Farm re-annotated",
            "Teacher-student RDB-Net with APL, HFPN and cascade head",
            "COCO mAP; paper table reports 54.5 on Bayonet-Drivers at 30% labeled",
            "Open paper/PMC; local code/checkpoint reproduction not established",
            "Small-object pyramid relevant; pseudo-labeling blocked pending approved data",
        ],
        [
            "GM-YOLOv7 + RST-Net (Applied Sciences 2024)",
            "Self-built seatbelt dataset",
            "Ghost/G-ELAN/Mish localization; pruned ResNet + Triplet Attention",
            "Paper RST-Net accuracy 98.25%, not Macro-F1",
            "Open article; no official code/checkpoint verified",
            "Torso ROI/attention only after three-state labels are approved",
        ],
        [
            "Overhead YOLOv7 seatbelt (ACS 2025)",
            "Day/night overhead traffic; four box classes",
            "YOLOv7, augmentation, anchor optimization; 640 input, 100 epochs",
            "Paper mAP50 97.46%, F1 95.37%, seatbelt AP 93.40%",
            "CC BY 4.0 paper; generic YOLOv7 code, no task checkpoint/dataset",
            "Relevant roadside geometry; Roadwatch lacks same-camera labeled data",
        ],
    ]
    literature = f"""# V9 scientific literature review

All values below are `REPORTED_IN_PAPER`. None is a Roadwatch result.

{_table(["Work", "Dataset/task", "Architecture/method", "Metric scope", "Code/checkpoint", "Roadwatch use"], literature_rows)}

Primary sources: [DualScope](https://ojs.aaai.org/index.php/AAAI/article/view/37812),
[Multi-Task Fusion](https://pmc.ncbi.nlm.nih.gov/articles/PMC12610507/),
[DriveCLIP](https://github.com/zahid-isu/DriveCLIP),
[DGMB-Net](https://pmc.ncbi.nlm.nih.gov/articles/PMC10935380/),
[RST-Net](https://www.mdpi.com/2076-3417/14/8/3346), and
[overhead YOLOv7](https://ph.pollub.pl/index.php/acs/article/view/7594).

Transferable hypotheses are behavior-centric crops, explicit motion differences, multi-scale temporal
context, small-object feature pyramids, and reliability-aware fusion. Each requires a Roadwatch ablation.
"""

    pending = seatbelt["pending_uncertain_review_manifest"]
    data_audit = f"""# V9 data and label quality audit

Scope is development-only: `{", ".join(diagnostics["development_subjects"])}`. Protected holdouts
`{", ".join(diagnostics["protected_holdouts"])}` were not accessed. Ground truth and event matching
were unchanged.

## Phone evidence

- The same 13 FN IDs occur in the evidence, detector trace and original-video ROI manifest.
- {fn["summary"]["decoded_detector_samples"]} original-video frames were decoded; only
  {fn["summary"]["samples_with_phone_box"]} have a phone proposal
  ({_pct(fn["summary"]["sampled_phone_box_coverage"])}).
- No FN has a verified occupant identity. ROI-level association is not an occupant track.
- Visual descriptions are inherited engineering review, not new label approval.

## Seatbelt labels

- Pending review: {pending["samples"]} candidates across {pending["unique_effective_groups"]} groups.
- Review states: `{pending["review_status"]}`; automatic training labels:
  {pending["automatic_training_labels_present"]}; occupant roles: `{pending["occupant_roles"]}`.
- Legacy UNCERTAIN queue: {seatbelt["legacy_uncertain_error_queue"]["samples"]} samples; locally
  reviewable ROIs: {seatbelt["legacy_uncertain_error_queue"]["roi_available_locally"]}.

Fine-tuning is `{seatbelt["fine_tuning_status"]}`. Invisible belt evidence remains
`UNCERTAIN_OR_OCCLUDED`; it is never converted automatically to `UNFASTENED`.
"""

    fn_rows = []
    for item in fn["cases"]:
        action = item["action"].split("/")[-1]
        t0 = item["action_probability"]["T0_max"]
        t1 = item["action_probability"]["T1_dense_max"]
        fn_rows.append(
            [
                item["case_id"],
                action,
                f"{item['duration_seconds']:.2f}",
                item["detector"]["samples_with_phone_box"],
                _pct(item["detector"]["phone_box_coverage"]),
                f"{t0:.3f}" if t0 is not None else "—",
                f"{t1:.3f}" if t1 is not None else "—",
                item["root_cause"],
            ]
        )
    fn_report = f"""# V9 phone FN/FP root causes

{_table(["Case", "Action", "s", "Box frames", "Coverage", "T0 max", "T1 max", "Bottleneck"], fn_rows)}

Only {fn["summary"]["cases_with_any_phone_box"]}/13 FN cases have any proposal in the five-frame trace;
0/13 have a verified occupant ID. Dense sampling crosses the fixed 0.80 action gate for
{fn["summary"]["T1_cases_reaching_action_gate"]}/13 but matches only
{fn["summary"]["T1_cases_matched"]} event. Geometry cannot rescue a missing box.

Pose landmarks are available for {pose_available}/13 cases. Geometry is blocked by a missing phone box
for {geometry_blocked}/13 and remains unverified without an occupant track for {geometry_unverified}/13;
only {handheld_context} proposal has hand/face proximity consistent with handheld use.

The visual audit contains clear large phones at the ear, partial occlusion, ROI-edge placement, glare and
motion blur. Detector domain/representation failure dominates sampling density. The generated
`phone_fn_human_review_queue.csv` preserves every case for human adjudication.
"""

    representation_report = f"""# V9 representation experiments

No model was trained. Existing frozen experiments and new ROI/motion instrumentation were analyzed first.

- Subject silhouette: {representation["frozen_embedding_subject_silhouette"]:.4f}.
- Action silhouette: {representation["frozen_embedding_action_silhouette"]:.4f}.
- Subject/action separability ratio: {representation["subject_to_action_separability_ratio"]:.2f}×.
- Dense result: {representation["dense_sampling_result"]}
- CLIP comparison: `{representation["clip_feature_comparison_status"]}`.

Frozen features retain much more subject identity than action structure. The next legitimate experiment is
a hash-registered comparison of full ROI, face/hand/torso crops, RGB differences and CLIP features under
the same LOSO split, reporting clip Macro-F1 separately from event F1.
"""

    fusion_rows = [
        [
            row["candidate"],
            row["true_positives"],
            row["false_positives"],
            row["false_negatives"],
            _pct(row["precision"]),
            _pct(row["recall"]),
            _pct(row["f1"]),
            f"{row['false_alarms_per_minute']:.4f}",
            f"{row['delta_tp']:+d}/{row['delta_fp']:+d}/{row['delta_fn']:+d}",
        ]
        for row in fusion
    ]
    event_table = _table(
        ["Experiment", "TP", "FP", "FN", "P", "R", "F1", "FA/min", "ΔTP/ΔFP/ΔFN"],
        fusion_rows,
    )
    fusion_report = f"""# V9 fusion ablations

{event_table}

T1/T3 recover one event but add 9/7 false positives versus the controlled comparator. T2–T5 do not
improve F1. The post-hoc row remains ineligible. Missing camera views are excluded from the denominator;
secondary cameras, pose and negative evidence cannot create an alert without physical and driver context.
"""

    seatbelt_rows = [
        [
            item["id"],
            _pct(item["macro_f1"]),
            _pct(item["balanced_accuracy"]),
            _pct(item["uncertain_recall"]),
            _pct(item["false_unfastened_rate"]),
            _pct(item["abstention_rate"]),
        ]
        for item in seatbelt["frozen_variants"]
    ]
    seatbelt_report = f"""# V9 seatbelt experiments

{_table(["Variant", "Macro-F1", "Balanced accuracy", "UNCERTAIN recall", "False-unfastened", "Abstention"], seatbelt_rows)}

Best measured development Macro-F1 is {_pct(seatbelt["best_development_macro_f1"])}. Of 25 legacy
UNCERTAIN failures, {seatbelt["legacy_uncertain_error_queue"]["high_confidence_wrong_at_0_90"]} are
confidently wrong at ≥0.90. Calibration alone cannot repair the missing class. Torso ROI, Triplet Attention
and hard-negative fine-tuning remain blocked by unapproved labels. No weights were changed.
"""

    event_report = f"""# V9 event metrics comparison

{event_table}

No eligible development result meets P/R/F1 ≥85%, FA/min ≤0.15 and SAFE recall ≥85%. The controlled
binary comparator remains TP=16, FP=4, FN=13, P=80.00%, R=55.17%, F1=65.31%, FA/min=0.1813.
Independent validation is absent; no development value is a production claim.
"""

    gpu_rows = [
        [
            item["id"],
            item["images"],
            f"{item['detector_call_fps']:.2f}",
            f"{item['p50_ms']:.2f}",
            f"{item['p95_ms']:.2f}",
        ]
        for item in gpu["variants"]
    ]
    runtime_report = f"""# V9 runtime benchmark

{_table(["Frozen T4 variant", "Images", "Detector FPS", "p50 ms", "p95 ms"], gpu_rows)}

The probe used one Tesla T4 device (`device=0`) and 64 legacy validation images without labels. It measures
detector calls, not full-system FPS. Local full-backend processing is 1.119 FPS on different inputs/hardware,
so no CPU/GPU speedup is claimed. V9 diagnostics add offline analysis only.
"""

    final_report = f"""# Roadwatch V9 final scientific report

V9 completed the prerequisite investigation and reproducible instrumentation; it did not fine-tune or
approve a model. Only {fn["summary"]["cases_with_any_phone_box"]}/13 FN events have any sampled phone box,
and none has verified occupant identity. Sampling density is downstream of this dominant failure.

Frozen embeddings separate subject identity {representation["subject_to_action_separability_ratio"]:.2f}×
more strongly than action. Dense sampling recovered one long event but raised false alarms. Seatbelt
development Macro-F1 peaks at {_pct(seatbelt["best_development_macro_f1"])}; 500 candidates remain pending.

- Event P/R/F1 ≥85%: **not achieved**.
- FA/min ≤0.15: **not established by an eligible candidate**.
- SAFE_DRIVING recall ≥85%: **not achieved**.
- Seatbelt Macro-F1 ≥85%: **not achieved**.
- Independent generalization: **not established**.
- Production approval: **false**.

The next model-changing run requires human-approved labels and a frozen representation protocol. Prioritize
a driver-centric phone detector, measure box recall and occupant association, then evaluate fusion. Keep
`gZ-37` and `gE-28` sealed until model, threshold and event policy are frozen.
"""

    reports = {
        "01_LITERATURE_REVIEW.md": literature,
        "02_DATA_LABEL_QUALITY_AUDIT.md": data_audit,
        "03_FN_FP_ROOT_CAUSES.md": fn_report,
        "04_REPRESENTATION_EXPERIMENTS.md": representation_report,
        "05_FUSION_ABLATIONS.md": fusion_report,
        "06_SEATBELT_EXPERIMENTS.md": seatbelt_report,
        "07_EVENT_METRICS_COMPARISON.md": event_report,
        "08_RUNTIME_BENCHMARK.md": runtime_report,
        "10_FINAL_SCIENTIFIC_REPORT.md": final_report,
    }
    output.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, content in reports.items():
        path = output / name
        _write(path, content)
        paths.append(path)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    print(json.dumps([str(path) for path in render(args.output)], indent=2))


if __name__ == "__main__":
    main()
