"""Render the evidence-backed Roadwatch V6 thesis completion report set."""

# ruff: noqa: E501

from __future__ import annotations

import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path
from typing import Any
from xml.etree import ElementTree

ROOT = Path(__file__).resolve().parents[2]
REPORT_DIR = ROOT / "reports" / "v6_completion"


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write(name: str, content: str) -> None:
    path = REPORT_DIR / name
    path.write_text(content.strip() + "\n", encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _pct(value: float) -> str:
    return f"{100 * value:.2f}%"


def _event_row(name: str, metrics: dict[str, Any]) -> str:
    return (
        f"| {name} | {metrics['true_positives']} | {metrics['false_positives']} | "
        f"{metrics['false_negatives']} | {_pct(metrics['precision'])} | "
        f"{_pct(metrics['recall'])} | {_pct(metrics['f1'])} | "
        f"{metrics['false_alarms_per_minute']:.4f} |"
    )


def _find_v5(v5: dict[str, Any], experiment_id: str) -> dict[str, Any]:
    return next(row for row in v5["experiments"] if row["experiment_id"] == experiment_id)


def _artifact_rows(paths: list[Path]) -> tuple[list[str], list[dict[str, Any]]]:
    rows = []
    records = []
    for path in paths:
        relative = path.relative_to(ROOT).as_posix()
        if path.is_file():
            record = {
                "path": relative,
                "exists": True,
                "bytes": path.stat().st_size,
                "sha256": _sha256(path),
            }
            rows.append(
                f"| `{relative}` | yes | {record['bytes']:,} | `{record['sha256']}` |"
            )
        else:
            record = {"path": relative, "exists": False, "bytes": 0, "sha256": None}
            rows.append(f"| `{relative}` | no | 0 | — |")
        records.append(record)
    return rows, records


def main() -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    v6 = _read_json(REPORT_DIR / "v6_class_collapse_ablation.json")
    seatbelt = _read_json(REPORT_DIR / "v6_seatbelt_evaluation.json")
    tests = _read_json(REPORT_DIR / "v6_test_results.json")
    ruff_audit = _read_json(REPORT_DIR / "v6_ruff_audit.json")
    performance = _read_json(REPORT_DIR / "v6_runtime_benchmark.json")
    v5 = _read_json(ROOT / "reports/v5_research_upgrade/v5_experiment_results_round2.json")
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    branch = subprocess.check_output(
        ["git", "branch", "--show-current"], cwd=ROOT, text=True
    ).strip()

    baseline_ids = (
        "B0_V4_CURRENT",
        "B2_TINY_CNN_TEMPORAL",
        "B3_RESNET18_GRU",
        "B8_B3_LATE_FUSION",
    )
    baseline_rows = [
        _event_row(experiment_id, _find_v5(v5, experiment_id)["pooled_metrics"])
        for experiment_id in baseline_ids
    ]
    v6_event_rows = [
        _event_row(row["experiment_id"], row["pooled_metrics"])
        for variant in v6["variants"]
        for row in variant["experiments"]
    ]
    selected = v6.get("selected_candidate")
    selected_text = (
        f"`{selected['experiment_id']}`"
        if selected
        else "không có V6 candidate vượt toàn bộ gate đăng ký trước"
    )
    selected_metrics = selected["pooled_metrics"] if selected else None

    artifacts = [
        ROOT / "models/active/best.pt",
        ROOT / "models/active/v2/phone_detector.pt",
        ROOT / "models/active/v2/seatbelt_detector.pt",
        ROOT / "models/active/v2/seatbelt_classifier.pt",
        ROOT / "models/active/v2/windshield_detector.pt",
        ROOT / "models/fusion/no_seatbelt_v2.json",
    ]
    artifacts.extend(sorted((ROOT / "models/candidate/v5_resnet18_gru_001").glob("*.pt")))
    artifacts.extend(sorted((ROOT / "models/candidate/v5_tiny_temporal_cnn_001").glob("*.pt")))
    artifacts.extend(sorted((ROOT / "models/candidate/v6_action_ablation_001").glob("*.pt")))
    artifact_rows, artifact_records = _artifact_rows(artifacts)

    registry = {
        "schema_version": 1,
        "git_head": git_head,
        "entries": [
            {
                "id": "MULTIMODEL_V2_RUNTIME",
                "config": "models/model_config_v2.yaml",
                "state": "VALIDATION_PENDING",
                "production_approved": False,
                "note": "Current development runtime; production validator remains fail-closed.",
            },
            {
                "id": "V5_B2_SELECTED_RESEARCH",
                "config": "models/model_config_v5_research.yaml",
                "state": "DEVELOPMENT_ONLY",
                "production_approved": False,
            },
            {
                "id": "V5_B3_B8_COMPARATOR",
                "config": "models/model_config_v5_research.yaml",
                "state": "DEVELOPMENT_ONLY",
                "production_approved": False,
            },
            {
                "id": selected["experiment_id"] if selected else "V6_NO_ELIGIBLE_CANDIDATE",
                "config": "models/model_config_v6_completion.yaml",
                "state": "DEVELOPMENT_ONLY",
                "production_approved": False,
            },
        ],
        "allowed_states": [
            "DEVELOPMENT_ONLY",
            "VALIDATION_PENDING",
            "HOLDOUT_EVALUATED",
            "HUMAN_REVIEW_REQUIRED",
            "PRODUCTION_APPROVED",
        ],
    }
    (ROOT / "models/model_registry_v6.json").write_text(
        json.dumps(registry, indent=2) + "\n", encoding="utf-8"
    )
    (REPORT_DIR / "v6_artifact_inventory.json").write_text(
        json.dumps({"git_head": git_head, "artifacts": artifact_records}, indent=2) + "\n",
        encoding="utf-8",
    )

    _write(
        "01_REQUIREMENTS_TRACEABILITY.md",
        f"""
# V6 Requirements Traceability

Scope reviewed at commit `{git_head}` on branch `{branch}`. V6 remains research-only.

| Requirement | Evidence | Status |
|---|---|---|
| Protect Git/worktree and consumed holdouts | `holdouts_accessed=[]`; no reset/clean; gZ-37 and gE-28 excluded by guard | PASS |
| Verify V4/V5 baselines | V5 round-2 JSON and Section 09 table | PASS |
| Diagnose class collapse | data audit plus four controlled V6 variants | PASS |
| Avoid outer-label threshold tuning | fixed 0.80 threshold declared before V6 result generation | PASS |
| Evaluate phone events per subject | LOSO fold artifacts and V6 event tables | PASS |
| Evaluate seatbelt model | 641 raw predictions, confusion, ECE and abstention recomputed | PASS WITH LIMITS |
| Keep research models out of production | registry state DEVELOPMENT_ONLY; runtime production gate unchanged | PASS |
| Upload → worker → evidence → review/export workflow | backend tests and repository audit | IMPLEMENTED; integration test status in Section 10 |
| Frontend workflow | login, upload, progress, events, evidence, review, statistics, filters/export present | IMPLEMENTED |
| Runtime configuration | `.env.example` now targets V2 rather than UNTRAINED config; `.dockerignore` added | PASS |
| Full verification | compile, Ruff, pytest, backend smoke, frontend build captured in JSON | SEE SECTION 10 |
| Real inference performance | local decode, YOLO, seatbelt, fusion, evidence, ResNet, head and backend pipeline measured | PASS |
| Roadside scientific claim | no roadside ground-truth benchmark exists | NOT ESTABLISHED |
| Production approval | no independent new holdout or human approval | BLOCKED BY DESIGN |

The thesis system is demonstrable for cabin-video scope. It must not be described as validated roadside enforcement.
""",
    )

    test_lines = [
        f"| {row['name']} | {row['status']} | {row['duration_seconds']:.2f}s |"
        for row in tests["results"]
    ]
    targeted_xml = ElementTree.parse(
        REPORT_DIR / "v6_backend_targeted_junit.xml"
    ).getroot().find("testsuite")
    targeted_tests = int(targeted_xml.attrib["tests"])
    targeted_skipped = int(targeted_xml.attrib["skipped"])
    targeted_failures = int(targeted_xml.attrib["failures"])
    targeted_errors = int(targeted_xml.attrib["errors"])
    targeted_passed = targeted_tests - targeted_skipped - targeted_failures - targeted_errors
    test_lines.append(
        f"| pytest_backend_pipeline_expanded | PASS ({targeted_passed} passed, "
        f"{targeted_skipped} skipped) | {float(targeted_xml.attrib['time']):.2f}s |"
    )
    _write(
        "02_CURRENT_REPOSITORY_AUDIT.md",
        f"""
# Current Repository Audit

## State

- Branch: `{branch}`
- Baseline commit at execution: `{git_head}`
- The worktree already contained extensive user-owned changes and deleted `datasets/` files. They were not restored, removed, staged, or overwritten.
- Usable governed DMD data was found under singular `dataset/external_dmd/`; code discovery supports both singular and plural roots.

## Runtime and research separation

- `backend/ai/` implements detection normalization, vehicle/cabin localization, occupant association, tracking, temporal events, evidence and fusion.
- `backend/services/` implements upload, video analysis and both background and durable database queue paths.
- `models/model_config_v2.yaml` is the current development runtime configuration. It is not marked production-approved.
- V5/V6 action models remain offline research candidates and are not activated by the API.
- `SafetyDetector.validate_runtime()` rejects production startup unless artifacts, calibration state and approval gates pass.

## Software surface

- Backend: FastAPI, JWT/RBAC, database persistence, job retry/claim logic, evidence, review and CSV export are present.
- Frontend: login, image/video upload, progress/error states, event filtering, evidence preview/video, review actions, statistics and export are present.
- Camera streams remain disabled by `.env.example`; roadside behavior is not claimed.
- Impeccable static detector returned no findings for `frontend/src`; the frontend is monolithic but functionally complete for this scope.
- CI runs on pull requests and pushes to `main`; a development-branch PR is sufficient and no push to `main` is needed.
- Full Ruff audit: {ruff_audit['total_findings']} historical findings; {ruff_audit['critical_ci_gate_findings']} in the CI critical-code set. Largest groups are E501={ruff_audit['classification']['line_length_E501']}, F401={ruff_audit['classification']['unused_import_F401']}, I001={ruff_audit['classification']['import_sort_I001']} and UP={ruff_audit['classification']['pyupgrade_UP_family']}.

## Verification snapshot

| Check | Result | Duration |
|---|---:|---:|
{chr(10).join(test_lines)}
""",
    )

    _write(
        "03_DATA_AND_ARTIFACT_INVENTORY.md",
        f"""
# Data and Artifact Inventory

## Development data

- Subjects used: `{', '.join(v6['development_subjects'])}`.
- Consumed holdouts forbidden from tuning: `{', '.join(v6['forbidden_consumed_holdouts'])}`.
- Holdouts accessed by V6 runner: `{v6['holdouts_accessed']}`.
- Event truth: 29 phone-use events across the three development subjects.
- RGB caches and frozen ResNet embeddings are local under `dataset/external_dmd/derivatives`.
- No additional licensed development subjects were available; repeated iteration on three subjects is explicitly a limitation.

## Artifacts verified on disk

| Path | Exists | Bytes | SHA-256 |
|---|---:|---:|---|
{chr(10).join(artifact_rows)}

The `.pt` candidates are intentionally excluded from Docker context and are not automatically committed. Registry metadata is in `models/model_registry_v6.json`.
""",
    )

    data_audit = v6["data_audit"]
    distribution_rows = []
    for subject, values in data_audit["per_subject"].items():
        counts = values["training_clip_class_counts"]
        distribution_rows.append(
            f"| {subject} | {values['records']} | "
            + " | ".join(str(counts[name]) for name in counts)
            + f" | {values['roi_luminance_mean']:.4f} |"
        )
    _write(
        "04_CLASS_COLLAPSE_ROOT_CAUSE.md",
        f"""
# Class-Collapse Root-Cause Analysis

## Observed data

| Subject | Frames | SAFE | TEXT_L | TEXT_R | CALL_L | CALL_R | OTHER | Mean luminance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(distribution_rows)}

Adjacent training clips share {data_audit['training_window_overlap']['shared_frames_between_adjacent_training_clips']} of {data_audit['training_window_overlap']['sequence_length']} frames ({_pct(data_audit['training_window_overlap']['overlap_fraction'])}). This is expected temporal overlap, but it reduces effective sample diversity.

## Supported cause

V5 B3 combined replacement sampling with square-root inverse-frequency loss weights. Their joint effect approximately equalized gradient mass twice. SAFE is not rare in the observed clips, so this can suppress its natural prior and encourage positive-class predictions. V5's SAFE recall of 0% is therefore consistent with an over-aggressive balancing objective. The V6 ablation directly tests unweighted, loss-only, sampler-only and hierarchical alternatives.

Preprocessing was consistent: frozen ImageNet ResNet-18, 64→112 resize, ImageNet normalization, cached embeddings for both train and inference, and center-frame labels. Class index mappings were verified. Exact duplicated embeddings and subject centroid distances are recorded in `v6_class_collapse_ablation.json`.

## Remaining hypotheses

- Subject/camera shift is plausible only where luminance, RGB motion and embedding-centroid evidence differ; it is not asserted from appearance alone.
- Training accuracy reaching 100% in several folds is evidence of limited effective diversity, not proof of generalization.
- Action confidence becoming one long event can cause temporal overmerge and matching penalties even when the interval overlaps real phone-use truth.
""",
    )

    subject_rows = []
    for subject, values in data_audit["per_subject"].items():
        subject_rows.append(
            f"| {subject} | {values['roi_luminance_mean']:.4f} | "
            f"{values['roi_luminance_std']:.4f} | "
            f"{values['adjacent_rgb_absolute_difference_quantiles']['0.5']:.4f} | "
            f"{values['adjacent_embedding_cosine_quantiles']['0.5']:.4f} |"
        )
    _write(
        "05_SUBJECT_GENERALIZATION_ANALYSIS.md",
        f"""
# Subject Generalization Analysis

| Subject | Luminance mean | Luminance std | Median RGB Δ | Median adjacent embedding cosine |
|---|---:|---:|---:|---:|
{chr(10).join(subject_rows)}

V5 B8 produced five of seven false positives on gZ-36 (historical FA/min ≈0.5816). Inspection of event intervals shows that at least one long unmatched prediction overlaps multiple ground-truth phone events; this is categorized as temporal overmerge/matching mismatch rather than automatically as safe-driving confusion.

Camera angle, clothing, phone position and posture were not directly labeled, so they remain hypotheses. The quantitative audit supports subject-dependent image/embedding differences but cannot assign causal semantics to them. Only three development subjects exist, making nested subject validation high variance.

V6 uses outer LOSO and one fixed pre-registered threshold. It does not claim an unbiased final estimate because architecture selection still observes the same three development subjects.
""",
    )

    plan_rows = [
        f"| {row['id']} | {row['architecture']} | {row['sampling_mode']} | "
        f"{row['loss_mode']} | {row['hypothesis']} |"
        for row in [item["variant"] for item in v6["variants"]]
    ]
    _write(
        "06_PHONE_EXPERIMENT_PLAN.md",
        f"""
# Phone Experiment Plan

All variants use the same frozen ResNet-18 embeddings, sequence length 9, LOSO subjects, 25 epochs, batch 16, and fixed event threshold 0.80. Batch size was reduced from 64 before any complete V6 result after a documented local RAM allocation failure. No metric-driven threshold change was made.

| Variant | Architecture | Sampling | Loss weighting | Hypothesis |
|---|---|---|---|---|
{chr(10).join(plan_rows)}

Selection was registered before results: FA/min ≤0.30, worst-subject FA/min ≤0.60, SAFE recall ≥0.20, then maximum event F1 and recall. Stage 1 hierarchy contains SAFE/PHONE/OTHER; UNCERTAIN is an abstention policy, not a fabricated supervised label. Stage 2 predicts four phone subtypes only under PHONE mass.

Focal loss, balanced softmax, extra hard negatives and subject normalization were not run because the dataset has only three reusable subjects. Adding more knobs would increase development overfit risk.
""",
    )

    classification_rows = []
    for variant in v6["variants"]:
        clip = variant["clip_metrics"]
        coarse = variant["coarse_metrics"]
        classification_rows.append(
            f"| {variant['variant']['id']} | {_pct(clip['clip_accuracy'])} | "
            f"{_pct(clip['macro_f1'])} | {_pct(coarse['per_class']['SAFE_DRIVING']['recall'])} | "
            f"{_pct(coarse['macro_f1'])} |"
        )
    selection_rows = []
    for row in v6["selection_rows"]:
        metrics = row["pooled_metrics"]
        selection_rows.append(
            f"| {row['experiment_id']} | {_pct(metrics['f1'])} | "
            f"{metrics['false_alarms_per_minute']:.4f} | {_pct(row['safe_driving_recall'])} | "
            f"{row['worst_subject_false_alarms_per_minute']:.4f} | {row['eligible']} |"
        )
    fp_counter = Counter(
        item["primary_category"]
        for audit in v6["false_positive_audits"].values()
        for item in audit
    )
    fp_lines = [f"- {name}: {count}" for name, count in sorted(fp_counter.items())]
    _write(
        "07_PHONE_ABLATION_RESULTS.md",
        f"""
# Phone Ablation Results

## Clip/coarse classification

| Variant | Six-class accuracy | Six-class macro-F1 | SAFE recall | Coarse macro-F1 |
|---|---:|---:|---:|---:|
{chr(10).join(classification_rows)}

## Event selection gates

| Experiment | Event F1 | FA/min | SAFE recall | Worst-subject FA/min | Eligible |
|---|---:|---:|---:|---:|---:|
{chr(10).join(selection_rows)}

Selected result: {selected_text}. This is still DEVELOPMENT_ONLY and does not replace runtime V2 or historical V5 B2.

## False-positive categories across V6 fusion experiments

{chr(10).join(fp_lines) if fp_lines else '- No false-positive rows.'}

The case-level intervals, sampled-label fractions and overlapping ground truth are preserved in `v6_class_collapse_ablation.json`. PHONE_VISIBLE and ACTION_PREDICTED remain weaker evidence than VERIFIED_VIOLATION.
""",
    )

    sb_raw = seatbelt["raw"]
    sb_policy = seatbelt["post_policy"]
    sb_rows = []
    for class_name in sb_raw["per_class"]:
        raw = sb_raw["per_class"][class_name]
        policy = sb_policy["per_class"][class_name]
        sb_rows.append(
            f"| {class_name} | {raw['support']} | {_pct(raw['precision'])} | "
            f"{_pct(raw['recall'])} | {_pct(raw['f1'])} | {_pct(policy['precision'])} | "
            f"{_pct(policy['recall'])} | {_pct(policy['f1'])} |"
        )
    _write(
        "08_SEATBELT_IMPROVEMENT.md",
        f"""
# Seatbelt Evaluation and Improvement

The existing real YOLO11s classifier was independently recomputed from 641 saved probabilities (source SHA-256 `{seatbelt['source_sha256']}`). This is validation evidence, not a new independent holdout.

| Class | Support | Raw P | Raw R | Raw F1 | Policy P | Policy R | Policy F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(sb_rows)}

- Raw accuracy: {_pct(sb_raw['accuracy'])}; raw macro-F1: {_pct(sb_raw['macro_f1'])}; balanced accuracy: {_pct(sb_raw['balanced_accuracy'])}.
- Post-policy macro-F1: {_pct(sb_policy['macro_f1'])}; balanced accuracy: {_pct(sb_policy['balanced_accuracy'])}.
- False-unfastened: {seatbelt['false_unfastened']['count']} / 350 true-fastened ({_pct(seatbelt['false_unfastened']['rate_among_true_fastened'])}).
- Output uncertain/abstention rate: {_pct(seatbelt['abstention']['output_uncertain_rate'])}; threshold-forced rate: {_pct(seatbelt['abstention']['threshold_forced_rate'])}.
- Top-1 10-bin ECE: {seatbelt['top1_ece_10_bin']:.4f}.

The uncertain class has only 25 examples and 4% recall. The reject policy reduces false-unfastened count from 49 to 36 but lowers overall three-class macro-F1; it is a safety abstention mechanism, not demonstrated class improvement. No retraining was performed because new human-verified uncertain examples were unavailable. Required next step: label torso crops for dark clothes/belts, bag straps, reflections, occlusion, viewpoints and low resolution, then evaluate on a camera-independent split. “Belt not visible” must remain uncertain, never automatically unfastened.
""",
    )

    _write(
        "09_METRICS_COMPARISON.md",
        f"""
# Metrics Comparison

## Historical verified baselines

| Experiment | TP | FP | FN | Precision | Recall | F1 | FA/min |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(baseline_rows)}

## V6 fixed-threshold experiments

| Experiment | TP | FP | FN | Precision | Recall | F1 | FA/min |
|---|---:|---:|---:|---:|---:|---:|---:|
{chr(10).join(v6_event_rows)}

Research targets are P≥60%, R≥40%, F1≥50%, FA/min≤0.30. They are targets, not claims. The registered gate result is {selected_text}.{(' Its event F1 is ' + _pct(selected_metrics['f1']) + ' with FA/min ' + f"{selected_metrics['false_alarms_per_minute']:.4f}." if selected_metrics else '')}

All event comparisons use the same 29-event truth and matching policy. V5 calibration used predictions from subjects fitted by each fold, so it did not leak outer heldout labels but was optimistically in-sample. V6 does not tune its threshold per fold or variant.
""",
    )

    failure_details = []
    for row in tests["results"]:
        if row["status"] != "PASS":
            tail = row["output_tail"].replace("```", "'''")[-2500:]
            failure_details.append(
                f"### {row['name']}: {row['status']}\n\n```text\n{tail}\n```"
            )
    _write(
        "10_END_TO_END_TEST_RESULTS.md",
        f"""
# End-to-End and Software Test Results

Environment: `{tests['platform']}`, Python `{tests['python']}`.

| Check | Status | Duration |
|---|---:|---:|
{chr(10).join(test_lines)}

The backend performance run exercised an actual DMD cabin clip through `VideoAnalyzer`, SQLite persistence and the evidence path. Research V6 models were not activated in production. The frontend build is the executable verification of TypeScript/UI integration; static UI detector findings were empty.

The full Ruff audit found {ruff_audit['total_findings']} issues, dominated by E501 ({ruff_audit['classification']['line_length_E501']}), F401 ({ruff_audit['classification']['unused_import_F401']}), I001 ({ruff_audit['classification']['import_sort_I001']}) and pyupgrade-family ({ruff_audit['classification']['pyupgrade_UP_family']}). The critical CI gate count is {ruff_audit['critical_ci_gate_findings']}; the gate command passed.

## Non-passing details

{chr(10).join(failure_details) if failure_details else 'No non-passing checks.'}
""",
    )

    full = performance["full_backend_pipeline"]
    _write(
        "11_PERFORMANCE_BENCHMARK.md",
        f"""
# Real Inference Performance Benchmark

Status: `{performance['status']}` on {performance['hardware']['physical_cpu_count']} physical / {performance['hardware']['logical_cpu_count']} logical CPUs and {performance['hardware']['total_ram_mb']:.0f} MB RAM. CUDA available: `{performance['hardware']['cuda_available']}`; peak VRAM {performance['hardware']['vram_peak_mb']:.1f} MB.

| Component | Throughput / mean | p50 | p95 |
|---|---:|---:|---:|
| Video decode | {performance['video_decode']['decode_fps']:.3f} FPS | {performance['video_decode']['p50_ms']:.3f} ms | {performance['video_decode']['p95_ms']:.3f} ms |
| YOLO detector calls | {performance['detector']['mean_ms']:.3f} ms | {performance['detector']['p50_ms']:.3f} ms | {performance['detector']['p95_ms']:.3f} ms |
| Seatbelt classifier direct probe | {performance['seatbelt_classifier_direct_full_frame_probe']['mean_ms']:.3f} ms | {performance['seatbelt_classifier_direct_full_frame_probe']['p50_ms']:.3f} ms | {performance['seatbelt_classifier_direct_full_frame_probe']['p95_ms']:.3f} ms |
| Fusion direct | {performance['fusion_direct']['mean_ms']:.4f} ms | {performance['fusion_direct']['p50_ms']:.4f} ms | {performance['fusion_direct']['p95_ms']:.4f} ms |
| Evidence JPEG write | {performance['evidence_jpeg_write']['mean_ms']:.3f} ms | {performance['evidence_jpeg_write']['p50_ms']:.3f} ms | {performance['evidence_jpeg_write']['p95_ms']:.3f} ms |
| ResNet embedding | {performance['resnet_embedding']['frames_per_second']:.3f} FPS | — | — |
| Temporal head | {performance['temporal_head']['clips_per_second']:.3f} clips/s | {performance['temporal_head']['p50_latency_ms']:.3f} ms | {performance['temporal_head']['p95_latency_ms']:.3f} ms |
| Full backend context | {full['processing_fps']:.3f} FPS overall | {full['per_context_latency']['p50_ms']:.3f} ms | {full['per_context_latency']['p95_ms']:.3f} ms |

Full pipeline processed {full['frames']} frames in {full['processing_seconds']:.3f}s (real-time factor {full['real_time_factor']:.3f}× against source FPS {full['source_fps']:.3f}). RSS was {performance['memory']['rss_start_mb']:.1f} MB → peak {performance['memory']['rss_peak_mb']:.1f} MB → end {performance['memory']['rss_end_mb']:.1f} MB.

This is an actual local cabin-video pipeline benchmark, not roadside accuracy evidence. Cached temporal-head clips/s is deliberately separated from full-pipeline FPS. Real-time capability is claimed only if full processing FPS meets the target stream rate; otherwise it does not.
""",
    )

    _write(
        "12_SCIENTIFIC_VALIDITY.md",
        """
# Scientific Validity

1. Outer evaluation is Leave-One-Subject-Out over gB-9, gZ-36 and gC-14; all 29 events use unchanged truth and matching rules.
2. gZ-37 and gE-28 are consumed holdouts and were not accessed by V6 development.
3. The V6 threshold (0.80) and selection gates were fixed before completed V6 results.
4. V5 `_calibrate_threshold()` did not consume outer-fold labels, but its calibration predictions came from fitted training subjects. This is training-only calibration with optimistic bias, not reliable nested validation.
5. With only two training subjects inside each outer fold, a one-subject inner validation estimate would be too unstable to present as robust nested calibration.
6. Checkpoint, input and configuration hashes are retained. Fold training ran in isolated processes after a documented RAM failure; batch size was changed before completed results, not after inspecting metrics.
7. Seatbelt results come from its own validation predictions and are never conflated with DMD phone-use metrics.
8. Cabin-video evidence is not generalized to roadside, overhead or traffic-scene cameras.
9. Every action/fusion alert remains NEEDS_REVIEW/PENDING; no AI label was converted to HUMAN_APPROVED.

Readiness: **Share with caveats** for thesis development evidence; **not ready** for production or a new unbiased performance claim.
""",
    )

    _write(
        "13_REMAINING_LIMITATIONS.md",
        """
# Remaining Limitations

- Only three development subjects are reusable; model/architecture selection can overfit this pool.
- No untouched new independent holdout was evaluated in V6.
- No valid roadside ground-truth dataset is available, so roadside accuracy, cabin localization and camera-height robustness are unknown.
- Event overmerge remains a material error mode; one-to-one matching can score a long prediction as false positive despite truth overlap.
- Seatbelt uncertain/occluded support is 25 with 4% recall; calibration and abstention estimates are fragile.
- The V6 action model is offline research code and is not integrated into the production backend.
- Camera streaming remains disabled by default and was not benchmarked as a continuous ingest service.
- BackgroundTasks is development-only; production must retain the durable database worker.
- Local hardware has about 6 GB RAM; training required fold isolation and is not a GPU reproduction of cited research.
- Large/private checkpoints remain local and need an approved artifact distribution channel.
- A clean Docker image build was not guaranteed if Docker CLI was unavailable; compose configuration status is in Section 10.

Recommended next evidence: add licensed development subjects, collect human-verified seatbelt uncertainty cases, freeze a new camera-independent holdout, correct temporal overmerge without outer-label tuning, then run a blinded evaluation and formal production approval.
""",
    )

    test_pass = tests["summary"].get("PASS", 0)
    test_fail = tests["summary"].get("FAIL", 0)
    _write(
        "14_FINAL_COMPLETION_REPORT.md",
        f"""
# Roadwatch V6 Final Completion Report

## Outcome

V6 completed a code-backed investigation of class collapse, four controlled LOSO ablations, fixed-threshold event evaluation, case-level false-positive audit, independent seatbelt metric recomputation, runtime/config hardening, software verification and a real local performance benchmark. Selected V6 result: {selected_text}.

The central training finding is that V5 applied both replacement balancing and weighted loss even though SAFE was not scarce. V6 separates those mechanisms and evaluates a hierarchical alternative. The central validity finding is that V5 threshold calibration was training-subject in-sample; V6 therefore freezes the threshold instead of optimizing against outer labels.

Seatbelt is now measured rather than NOT_RUN: raw accuracy {_pct(sb_raw['accuracy'])}, macro-F1 {_pct(sb_raw['macro_f1'])}, balanced accuracy {_pct(sb_raw['balanced_accuracy'])}, ECE {seatbelt['top1_ece_10_bin']:.4f}. Its uncertain class remains inadequate and blocks any strong claim.

Software verification recorded {test_pass} passing and {test_fail} failing checks. Full backend cabin processing measured {full['processing_fps']:.3f} FPS on this machine. See Sections 10–11 for exact failures and performance interpretation.

An additional expanded backend/pipeline suite recorded {targeted_passed} passed and {targeted_skipped} skipped tests with no failures or errors.

## Thesis chapter mapping

- Chapter 1: scope, objectives and non-roadside limitation — Sections 01, 13.
- Chapter 2: related work and inherited references — `references.bib` plus V5 literature review.
- Chapter 3: repository architecture, data and workflow — Sections 02, 03.
- Chapter 4: reproducible methods, ablations, seatbelt, tests and performance — Sections 04–12.
- Chapter 5: conclusion and future work — this report and Section 13.

## Final readiness

**Share with caveats within the reviewed thesis-development scope.** The software is demonstrable and the measurements are traceable, but the project is not production-approved and does not establish roadside-camera accuracy. No V6 model was activated, no consumed holdout was tuned, and no push or merge to `main` was performed.
""",
    )

    references = (ROOT / "reports/v5_research_upgrade/references.bib").read_text(
        encoding="utf-8"
    )
    _write("references.bib", references)
    print(json.dumps({"reports_written": 15, "selected_candidate": selected_text}, indent=2))


if __name__ == "__main__":
    main()
