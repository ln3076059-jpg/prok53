"""Render the V15 research record from machine-readable artifacts."""

# ruff: noqa: E501

from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from pathlib import Path

from training.common import sha256_file
from training.dmd.v12_reports import read_native_log

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v15_research"


def read(name: str) -> dict:
    return json.loads((OUT / name).read_text(encoding="utf-8"))


def write(name: str, body: str) -> None:
    (OUT / name).write_text(body.strip() + "\n", encoding="utf-8")


def junit(name: str) -> dict:
    path = OUT / name
    if not path.exists():
        return {"tests": 0, "failures": 0, "errors": 0, "skipped": 0, "status": "NOT_RUN"}
    root = ET.parse(path).getroot()
    suite = root[0] if root.tag == "testsuites" and len(root) else root
    values = {key: int(float(suite.attrib.get(key, 0))) for key in ("tests", "failures", "errors", "skipped")}
    values["status"] = "PASS" if values["failures"] + values["errors"] == 0 else "FAIL"
    return values


def run() -> None:
    metrics = read("metrics_comparison.json")
    ledger = read("fn_gate_ledger.json")
    sliced = read("sliced_phone_diagnostic.json")
    data = read("dataset_manifest.json")
    registry = read("experiment_registry.json")
    targeted, full = junit("pytest_targeted.xml"), junit("pytest_full.xml")
    event = metrics["phone_event_controlled_29"]
    e3 = metrics["clip_selected_e3"]
    seatbelt = metrics["seatbelt"]
    stages = ledger["nonexclusive_failure_stage_counts"]

    write("01_FULL_REPOSITORY_AI_AUDIT.md", """# Full repository AI audit

Audit scope: the requested `backend/ai`, `training/dmd`, model configs, registry/API integration, frontend API use, evidence and review paths were read directly. There is no `configs/` directory. The runtime and research stacks share concepts but are separate implementations.

| Stage | Runtime implementation | Input → output / coordinates | Failure and activation state |
|---|---|---|---|
| Decode/context | `backend/services/video_analyzer.py` | uploaded video → BGR frames and vehicle/cabin contexts | rejects unsupported scope; analysis worker serializes inference |
| Detection | `backend/ai/detector.py` | HWC frame/crop → `NormalizedDetection` in input-pixel XYXY | lazy YOLO; thresholded by config; configured weights can be absent; candidate is development-only |
| Driver localization | `backend/ai/cabin.py`, `association.py` | cabin/person boxes → explicit role/confidence | geometry below confidence stays UNKNOWN; tracking never proves role |
| ROI | runtime crop helpers; research `behavior_roi.py`, `high_res_roi.py` | source XYXY → crop plus scale/padding provenance | research source-resolution path is not wired into production `VideoAnalyzer` |
| Pose/phone context | `backend/ai/auxiliary.py` | crop → COCO keypoints/context | missing weights/evidence stays unknown; mounted/static is non-use |
| Action | runtime has no fitted E3; research `online_candidate.py`, `action_model.py`, `causal_motion.py` | causal 64×64 RGB history → 444-D E3 → three scores | fitted head/scaler missing; normalized OvR sigmoid is explicitly uncalibrated |
| Association | `association.py`, `tracking.py` | detections/occupants → role and stable IDs | overlapping/low-confidence roles abstain; IoU tracker differs from configured ByteTrack vehicle path |
| Fusion | `backend/ai/fusion.py`; research `training/dmd/fusion.py` | detector/context/pose/temporal features → score | runtime may use a rule fallback unless calibrated fusion is required; research multiview code is not the same runtime path |
| Temporal event | `backend/ai/sequence.py`, `events.py` | timestamped observations → candidates | driver-only phone gate; strict unknown context becomes NEEDS_REVIEW; cooldown/gap/occlusion rules are config-driven |
| Evidence/review | `evidence.py`, API routes, `research_review.py` | event frames/traces → hashed files and review receipts | file traversal/duplicates rejected; human and AI provenance remain distinct |

The frontend does not run models; it starts analysis through FastAPI, polls job/event APIs, and presents evidence/review state. The active candidate config is `models/model_config_v4_candidate.yaml`: inherited unvalidated thresholds, `production_approved=false`, strict phone context, missing per-camera calibration and no locked independent validation. `model_config.yaml` and AI85 research configs represent different historical/candidate contracts, so their thresholds must not be mixed.

The largest duplication risk is `training/dmd/fusion.py` versus `backend/ai/fusion.py` and research `OnlineE3` versus the production `VideoAnalyzer`. A future model should be exported as one versioned artifact and consumed by the runtime adapter; it should not create a second event engine.
""")

    write("02_VERIFIED_SCIENTIFIC_LITERATURE.md", """# Verified scientific literature

The full structured matrix is `literature_matrix.json`; fields that could not be confirmed are marked `NOT VERIFIED`.

- [SAHI, ICIP 2022](https://arxiv.org/abs/2202.06934) provides the strongest direct fit: overlapping slices, original-coordinate remapping and post-processing address small phones. Its VisDrone/xView AP gains are external detection results.
- [DualScope, AAAI 2026](https://ojs.aaai.org/index.php/AAAI/article/view/37812) combines behavior-centric spatial distillation, fine motion and temporal pyramids. The supplied GitHub returned unavailable and the checkpoint page could not be authenticated, so license/reproducibility remain unverified.
- [DriveCLIP](https://github.com/zahid-isu/DriveCLIP) supports the frame-based framework. The README samples video frames and predicts frames, while linking separate VideoCLIP code; it is not evidence of a complete Roadwatch temporal event pipeline.
- [ActionFormer, ECCV 2022](https://www.ecva.net/papers/eccv_2022/papers_ECCV/papers/136640485.pdf) predicts categories and temporal boundaries from multi-scale features with local self-attention. It uses pre-extracted features and is too costly before a small causal boundary baseline.
- [Multi-Task Fusion, Sensors 2025](https://www.mdpi.com/1424-8220/25/21/6799) uses 3D ResNet18 plus BiLSTM across BODY/FACE/HANDS tasks. Missing views must remain missing evidence, not negative evidence.
- [MixStyle](https://arxiv.org/abs/2104.02008) and [Group DRO](https://arxiv.org/abs/1911.08731) target domain/group robustness. Group DRO is rejected now because only three repeatedly used development subjects are available and the method needs group labels, regularization and early stopping.
- [ByteTrack](https://github.com/FoundationVision/ByteTrack) can preserve low-score boxes through association, but a track ID does not establish the driver's role.
- [RST-Net seatbelt](https://www.mdpi.com/2076-3417/14/8/3346) supports torso localization, pruning and Triplet Attention. Its private/self-built data and image F1 cannot be transferred to Roadwatch. The [overhead YOLOv7 study](https://ph.pollub.pl/index.php/acs/article/view/7594) has a roadside geometry mismatch.
- The current [DMD repository](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset) says the 2026 release is real-car RGB only and CC BY-NC-ND 4.0. OpenLABEL/VCD annotations and BODY/FACE/HANDS exist, but local authorization still denies fitting/upload.
- [DGMB-Net](https://www.mdpi.com/1424-8220/24/5/1386) uses teacher-student semi-supervision, adaptive perceptual learning and a hierarchical feature pyramid on Bayonet-Drivers/State Farm. It is detection research from different camera/data conditions.

No paper classification accuracy, AP, mAP, MOTA or image F1 is reported as Roadwatch Event F1.
""")

    write("03_RESEARCH_GAP_AND_METHOD_SELECTION.md", """# Research gap and method selection

| Rank | Method | Root-cause fit | Reproducibility | Compute | Fair evaluation now | Decision |
|---:|---|---|---|---|---|---|
| 1 | SAHI-inspired fixed-checkpoint ROI tiling | direct: 11/13 specialist misses | high for geometry/code | low-medium inference | geometry only; AP blocked | KEEP engineering |
| 2 | FN gate ledger and abstention contract | direct: all 13 have unknown association | high | low | yes, diagnostic | KEEP |
| 3 | lightweight causal spatial-motion model | direct: 9/13 below action gate, 4 temporal risk | medium | medium | no fitting rights | BLOCKED |
| 4 | ByteTrack phone association | fragmentation/low confidence | high | medium | no role GT | BLOCKED |
| 5 | DriveCLIP/DualScope | action representation | partial/unverified assets | medium-high | no fitting rights | BLOCKED |
| 6 | Group DRO | worst-subject behavior | reproducible | medium | only 3 reused groups | REJECT now |

The three preregistered V15 experiments are in `experiment_plan.json`. V15-A changes only tiling; V15-B reconciles frozen evidence; V15-C adds regression coverage for abstention semantics. This order targets observed failures without claiming recovery from an FN-selected sample.
""")

    write("04_DATA_RIGHTS_AND_SPLIT_AUDIT.md", f"""# Data rights and split audit

The authoritative local record remains `RIGHTS_ACCEPTED={data['permissions']['RIGHTS_ACCEPTED']}`, `CANONICAL_ELIGIBLE={data['permissions']['CANONICAL_ELIGIBLE']}`, `TEMPORAL_DEVELOPMENT_ELIGIBLE={data['permissions']['TEMPORAL_DEVELOPMENT_ELIGIBLE']}`. `linear_head`, calibration, fusion, boundary/backbone fitting and upload all fail closed. V15 performed no fitting or upload.

`gB-9`, `gC-14`, and `gZ-36` are repeated development subjects. `gZ-37` and `gE-28` are protected consumed holdouts and were not opened. There is no independent holdout. Current official DMD licensing is CC BY-NC-ND 4.0; distribution of transformed derivatives is restricted. Local permission, source terms and institutional approval must all be satisfied before training or sharing.

Future valid evaluation needs session/video/subject deduplication, group-disjoint train/validation/test, a locked holdout, dataset hash, frozen checkpoint/threshold/matcher, and per-subject reporting. Current results are Level 1/2 development evidence only.
""")

    write("05_ERROR_ROOT_CAUSE_MATRIX.md", f"""# Error root-cause matrix

The case ledger in `fn_gate_ledger.json` preserves all 13 IDs and source hash. Categories are nonexclusive:

| Evidence-backed stage | Cases |
|---|---:|
| Specialist detector miss | {stages.get('SPECIALIST_DETECTOR_MISS', 0)} |
| Auxiliary candidate not admitted | {stages.get('AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE', 0)} |
| Driver association unknown | {stages.get('DRIVER_ASSOCIATION_UNKNOWN', 0)} |
| Action evidence below fixed gate | {stages.get('ACTION_EVIDENCE_BELOW_GATE', 0)} |
| Subsecond temporal risk | {stages.get('SUBSECOND_TEMPORAL_RISK', 0)} |
| Occlusion/localization failure | {stages.get('OCCLUSION_OR_LOCALIZATION_FAILURE', 0)} |

Primary frozen rejection was `PHONE_DETECTOR_NO_PROPOSAL` for {ledger['primary_gate_rejection_counts'].get('PHONE_DETECTOR_NO_PROPOSAL', 0)} cases and `PHONE_BOX_PRESENT_ACTION_UNDER_GATE` for {ledger['primary_gate_rejection_counts'].get('PHONE_BOX_PRESENT_ACTION_UNDER_GATE', 0)}. All 13 reviewed cases show a phone and all 13 lack verified driver association. Nine fail the action gate. These facts support a staged detector → association → action study; they do not estimate possible new FP, precision, AP or recovered TP.
""")

    write("06_SMALL_PHONE_DETECTION_STUDY.md", f"""# Small-phone detection study

V15 adds `training/dmd/sliced_phone.py`, a SAHI-inspired adapter around the existing `PhoneDetector` protocol. It covers an original-resolution ROI with overlapping tiles, converts tile-local XYXY boxes to source-frame coordinates, preserves checkpoint hash/confidence/source, and runs confidence-ordered single-class NMS. It accepts no ground truth or review metadata.

Synthetic 1280×720 geometry on ROI `[0,72,832,684]` with 384×384 tiles and 25% overlap produced {sliced['tile_count']} tiles. Thirty adapter-only runs measured p50/p95/p99 {sliced['latency_ms']['p50']:.3f}/{sliced['latency_ms']['p95']:.3f}/{sliced['latency_ms']['p99']:.3f} ms. Model inference was excluded. Tests verify boundary coverage, local-to-original mapping, overlap deduplication, empty output and invalid config.

Decision: KEEP engineering; accuracy BLOCKED. No complete positive/negative bbox benchmark exists, so phone AP50, mAP50-95, precision, recall and negative-frame FP remain null. The adapter is not connected to active production config.
""")

    write("07_ACTION_REPRESENTATION_STUDY.md", f"""# Action representation study

The inherited E3 development result remains Macro-F1 {e3['pooled']['macro_f1']:.4f}, balanced accuracy {e3['pooled']['balanced_accuracy']:.4f}, and SAFE/PHONE/OTHER recall {e3['pooled']['safe_recall']:.4f}/{e3['pooled']['phone_recall']:.4f}/{e3['pooled']['other_recall']:.4f}. Worst-subject SAFE recall is {e3['worst_subject']['safe_recall']:.4f}. Twenty-five probability-argmax disagreements remain part of the reconciliation record.

The code already has original-resolution ROI extraction, causal motion, causal running standardization and a strict NumericHead schema. The fitted E3 head/scaler is missing. No new action model was fitted, and synthetic coefficients were not used. DualScope-inspired spatial/motion features, frozen DriveCLIP embeddings and a small causal boundary head remain preregisterable future ablations after rights approval. SAFE collapse and worst-subject behavior remain unresolved.
""")

    write("08_ASSOCIATION_AND_FUSION_STUDY.md", """# Association and fusion study

Runtime association combines calibrated ROIs or cabin geometry with explicit confidence and returns `unknown` below threshold. The local tracker provides continuity but cannot infer driver role. All 13 FN cases lack verified occupant tracks, so driver-association accuracy is not measurable.

Runtime `LogisticFusion` accepts detector, vehicle, occupant, phone-context, pose, seatbelt and temporal features. When calibrated fusion is not required, it has a conservative rule fallback. Research multiview fusion is a separate path and must not silently replace runtime semantics. Missing FACE/HANDS remain unavailable, not negative evidence.

V15 retains raw candidates while unknown driver/handheld evidence remains UNKNOWN/NEEDS_REVIEW. Mounted phone/camera evidence is non-use. Decision: KEEP the abstention contract; BLOCK association/fusion accuracy claims until verified role GT and a fitted/calibrated artifact exist.
""")

    write("09_EVENT_LEVEL_VALIDATION.md", f"""# Event-level validation

No new V15 event replay was run because tiling has no validated detector accuracy and driver association/E3 artifacts are missing. Ground truth, matcher, thresholds, exposure and duplicate policy were unchanged.

The frozen development comparator remains TP={event['true_positives']}, FP={event['false_positives']}, FN={event['false_negatives']}, precision={event['precision']:.4f}, recall={event['recall']:.4f}, F1={event['f1']:.4f}, FA/min={event['false_alarms_per_minute']:.4f}. Recovered TP, lost TP, new FP, removed FP, short-event recall, fragmentation and emission latency are not newly measured. `event_match_records.json` records this non-run explicitly.

Candidate preservation is not event admission. Unknown evidence cannot be relabeled SAFE or a confirmed violation. Independent AI85 is not established.
""")

    write("10_SEATBELT_STUDY.md", f"""# Seatbelt study

The inherited best development Macro-F1 is {seatbelt['best_development_macro_f1']:.4f}; UNCERTAIN recall remains 0.04 in all reported variants that retain any uncertain predictions. The best historical S2 flip-TTA result has balanced accuracy 0.58396, false-unfastened rate 0.08 and abstention rate 0.054602.

The evidence supports a future two-stage model: first classify `OBSERVABLE` versus `UNCERTAIN_OR_OCCLUDED`; only then classify observable torso crops as FASTENED/UNFASTENED. RST-Net motivates torso ROI, pruning and Triplet Attention, while the overhead YOLOv7 paper has a material camera-domain mismatch.

All 500 review candidates remain pending. V15 did not approve labels, train, calibrate or change seatbelt runtime. Invisible belts are never UNFASTENED. Decision: BLOCKED pending authenticated human labels, rights and a group-disjoint split.
""")

    write("11_RUNTIME_AND_TEST_RESULTS.md", f"""# Runtime and test results

| Check | Result |
|---|---|
| Targeted pytest | {targeted['status']}: {targeted['tests']} tests, {targeted['failures']} failures, {targeted['errors']} errors |
| Full pytest | {full['status']}: {full['tests']} tests, {full['failures']} failures, {full['errors']} errors |
| Python compile | see `compile.log` |
| Ruff | see `ruff.log` |
| Frontend build | see `frontend_build.log` |
| Backend smoke | see `backend_smoke.log` |

The only new timing is synthetic adapter geometry: p50/p95/p99 {sliced['latency_ms']['p50']:.3f}/{sliced['latency_ms']['p95']:.3f}/{sliced['latency_ms']['p99']:.3f} ms for {sliced['tile_count']} tiles, excluding model inference. It is not detector FPS, offline throughput, live latency, or full-pipeline FPS. Full decode→evidence CPU/GPU/RAM/VRAM profiling was NOT RUN because no valid new candidate model exists.
""")

    write("12_FINAL_AI_LEAD_REVIEW.md", f"""# Final AI lead review

Decision: **RESEARCH ONLY — REJECT FOR PRODUCTION**.

V15 verified the literature, audited runtime/research data flow, reconciled every one of the 13 false negatives, implemented a fixed-checkpoint-compatible tiled phone adapter, and strengthened geometry/abstention regression coverage. The dominant observed failures overlap: {stages.get('SPECIALIST_DETECTOR_MISS', 0)} specialist misses, {stages.get('DRIVER_ASSOCIATION_UNKNOWN', 0)} unknown driver associations and {stages.get('ACTION_EVIDENCE_BELOW_GATE', 0)} action-gate failures. The adapter is scientifically relevant but has no valid accuracy benchmark yet.

No accuracy metric changed. Phone Event P/R/F1 remains 80.00%/55.17%/65.31% with 0.1813 FA/min. E3 Macro-F1 remains 56.39%, SAFE recall 60.38%, PHONE recall 66.78%, OTHER recall 43.26%, worst-subject SAFE recall 54.36%. Seatbelt Macro-F1 remains 58.52% and UNCERTAIN recall 4%. Independent AI85 is not established; `production_approved=false`.

Blockers are rights denial, absence of a complete positive/negative phone bbox benchmark, missing verified driver-role GT, missing fitted E3 head/scaler, 500 pending seatbelt labels and no locked independent holdout. The single highest-value next experiment is an authorized group-disjoint detection set containing positive and negative BODY frames with phone boxes and driver-role labels, then a fixed-checkpoint A0/A2/A3 comparison including AP, negative FP and model-inclusive latency.
""")

    test_results = {
        "targeted": targeted,
        "full": full,
        "compile": "PASS" if "PASS" in read_native_log(OUT / "compile.log") else "CHECK_LOG",
        "ruff": "PASS" if "All checks passed" in read_native_log(OUT / "ruff.log") else "CHECK_LOG",
        "frontend": "PASS" if "built in" in read_native_log(OUT / "frontend_build.log") else "CHECK_LOG",
        "backend": "PASS" if '"status": "PASS"' in read_native_log(OUT / "backend_smoke.log") else "CHECK_LOG",
    }
    write("test_results.json", json.dumps(test_results, indent=2))
    manifest = {
        path.relative_to(ROOT).as_posix(): {"sha256": sha256_file(path), "bytes": path.stat().st_size}
        for path in sorted(OUT.rglob("*"))
        if path.is_file() and path.name not in {"artifact_manifest.json", "validation.json"}
    }
    write("artifact_manifest.json", json.dumps(manifest, indent=2))
    validation = {
        "reports_present": all((OUT / f"{index:02d}_{name}.md").exists() for index, name in enumerate((
            "FULL_REPOSITORY_AI_AUDIT", "VERIFIED_SCIENTIFIC_LITERATURE", "RESEARCH_GAP_AND_METHOD_SELECTION",
            "DATA_RIGHTS_AND_SPLIT_AUDIT", "ERROR_ROOT_CAUSE_MATRIX", "SMALL_PHONE_DETECTION_STUDY",
            "ACTION_REPRESENTATION_STUDY", "ASSOCIATION_AND_FUSION_STUDY", "EVENT_LEVEL_VALIDATION",
            "SEATBELT_STUDY", "RUNTIME_AND_TEST_RESULTS", "FINAL_AI_LEAD_REVIEW"
        ), start=1)),
        "experiment_count": len(read("experiment_plan.json")["experiments"]),
        "training_performed": registry["training_performed"],
        "protected_holdouts_accessed": registry["protected_holdouts_accessed"],
        "production_approved": registry["production_approved"],
        "independent_ai85_established": False,
    }
    write("validation.json", json.dumps(validation, indent=2))


if __name__ == "__main__":
    run()
