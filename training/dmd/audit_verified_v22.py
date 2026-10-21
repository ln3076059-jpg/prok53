"""Recompute recovery artifacts from repository evidence and real diagnostic inference.

No fitting, threshold selection, data approval, protected-subject access, or upload.
Reconstructed historical evidence is explicitly distinct from online compatibility.
"""

from __future__ import annotations

import argparse
import ast
import csv
import json
import platform
import re
import subprocess
import time
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
import torch

from training.dmd.action_checkpoint import FEATURE_CONTRACT, sha256_file
from training.dmd.adapter import parse_dmd_openlabel
from training.dmd.run_verified_v22 import OUTPUT, ROOT, read_json, write_json
from training.dmd.v22_evaluation import confusion_metrics, metrics_from_raw

SUBJECTS = ("gB-9", "gC-14", "gZ-36")


def action_claim_audit(output: Path) -> dict:
    report = ROOT / "reports/v21_ai85_recovery/08_ACTION_CLASSIFICATION_METRICS.md"
    text = report.read_text(encoding="utf-8")
    matrix = []
    for label in ("SAFE_DRIVING", "PHONE_USE", "OTHER_ACTIVITY"):
        match = re.search(rf"^{label}\s+(\d+)\s+(\d+)\s+(\d+)\s+\d+\s*$", text, re.M)
        if not match:
            raise ValueError("Cannot locate published confusion matrix")
        matrix.append([int(v) for v in match.groups()])
    arithmetic = confusion_metrics(matrix)
    source = ROOT / "reports/v21_ai85_recovery/action_predictions.jsonl"
    rows = [
        json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()
    ]
    label_fields = {"label", "true_label", "ground_truth", "y_true"}
    labels_present = sum(bool(set(row) & label_fields) for row in rows)
    scores = Counter(
        tuple(row.get(k) for k in ("safe_driving", "phone_use", "other_activity")) for row in rows
    )
    audit = {
        "status": "ACTION_METRICS_NOT_REPRODUCIBLE",
        "scope": "RECOMPUTED_ARITHMETIC_FROM_PUBLISHED_MATRIX_ONLY",
        "source_report_sha256": sha256_file(report),
        "source_prediction_sha256": sha256_file(source),
        "reported_macro_f1": 0.8516,
        "matrix_arithmetic": arithmetic,
        "raw_prediction_rows": len(rows),
        "rows_with_ground_truth": labels_present,
        "distinct_score_triplets": len(scores),
        "score_triplet_counts": [{"scores": list(k), "count": v} for k, v in scores.most_common()],
        "reason": "No aligned true labels/sample provenance for the claimed hybrid 3600-sample experiment",
        "ai85_verified": False,
    }
    write_json(output / "action_confusion_matrix.json", {"scope": audit["scope"], **arithmetic})
    write_json(output / "action_metrics_verified.json", audit)
    return audit


def checkpoint_diagnostics(output: Path) -> dict:
    from training.dmd.v5_resnet_gru import EmbeddingClipDataset, SubjectEmbeddings
    from training.dmd.v6_action_hierarchy import (
        HierarchicalResNetGRUClassifier,
        predict_hierarchical,
    )
    from training.dmd.v7_action_models import BinaryGateResNetGRUClassifier, predict_v7

    torch.set_num_threads(1)
    extraction_path = ROOT / "reports/v5_research_upgrade/v5_experiment_results_round2.json"
    extraction = read_json(extraction_path)
    extraction_records = {r["subject_id"]: r for r in extraction["input_provenance_b3"]}
    rgb_records = {r["subject_id"]: r for r in extraction["input_provenance"]}
    diagnostics = {
        "scope": "OFFLINE_CENTERED_CACHED_FEATURE_LOSO_DEVELOPMENT_DIAGNOSTIC",
        "online_hybrid_metrics": False,
        "training_performed": False,
        "families": {},
        "checkpoints": [],
    }
    private = output / "action_diagnostics_private"
    private.mkdir(parents=True, exist_ok=True)
    for family, stem, model_dir, fold_dir in (
        ("V6", "hierarchical_loss_only", "v6_action_ablation_001", "v6_completion"),
        ("V7", "v7_d_binary_gate", "v7_research", "v7_optimization"),
    ):
        all_labels, all_predictions = [], []
        family_results = {}
        for subject in SUBJECTS:
            name = f"{stem}_loso_{subject}"
            checkpoint = ROOT / f"models/candidate/{model_dir}/{name}.pt"
            summary_path = ROOT / f"reports/{fold_dir}/folds/{name}.json"
            prediction_path = ROOT / f"reports/{fold_dir}/folds/{name}.npz"
            cache = (
                ROOT
                / f"dataset/external_dmd/derivatives/v5_resnet18_embeddings/v5_resnet18_{subject}.npz"
            )
            expected = read_json(summary_path)
            cache_evidence = extraction_records[subject]
            if sha256_file(cache) != cache_evidence["output_sha256"]:
                raise ValueError(f"HISTORICAL_FEATURE_CACHE_SHA_MISMATCH: {subject}")
            rgb_cache = ROOT / cache_evidence["rgb_cache_path"]
            if sha256_file(rgb_cache) != cache_evidence["rgb_cache_sha256"]:
                raise ValueError(f"HISTORICAL_RGB_CACHE_SHA_MISMATCH: {subject}")
            source_record = rgb_records[subject]
            for field, digest_field in (
                ("video_path", "video_sha256"),
                ("annotation_path", "annotation_sha256"),
            ):
                if sha256_file(ROOT / source_record[field]) != source_record[digest_field]:
                    raise ValueError(f"HISTORICAL_SOURCE_SHA_MISMATCH: {subject}")
            checkpoint_sha = sha256_file(checkpoint)
            if checkpoint_sha != expected["checkpoint_sha256"]:
                raise ValueError(f"HISTORICAL_CHECKPOINT_SHA_MISMATCH: {name}")
            data = torch.load(checkpoint, map_location="cpu", weights_only=True)
            train = list(data["training_subjects"])
            if (
                data["heldout_subject"] != subject
                or subject in train
                or set(train) != set(expected["training_subjects"])
            ):
                raise ValueError("HISTORICAL_SPLIT_MISMATCH")
            if set(train) - set(SUBJECTS):
                raise ValueError("UNAUTHORIZED_TRAIN_SUBJECT")
            model = (
                HierarchicalResNetGRUClassifier()
                if family == "V6"
                else BinaryGateResNetGRUClassifier()
            )
            model.load_state_dict(data["model_state_dict"], strict=True)
            embeddings = SubjectEmbeddings.load(subject, cache)
            with np.load(rgb_cache, allow_pickle=False) as original_rgb:
                if not np.array_equal(embeddings.labels, original_rgb["labels"]):
                    raise ValueError("RGB_AND_EMBEDDING_LABELS_DIFFER")
            dataset = EmbeddingClipDataset(
                [embeddings], sequence_length=9, stride=1, labeled_only=False
            )
            started = time.perf_counter()
            if family == "V6":
                result = predict_hierarchical(model, dataset, batch_size=16)
            else:
                result = predict_v7(model, dataset, architecture="binary_gate", batch_size=16)
            elapsed = time.perf_counter() - started
            valid = result["labels"] >= 0
            true_flat = result["labels"][valid]
            true_coarse = np.where(true_flat == 0, 0, np.where(true_flat == 5, 2, 1))
            probabilities = result["probabilities"][valid]
            coarse_scores = np.column_stack(
                [probabilities[:, 0], probabilities[:, 1:5].sum(1), probabilities[:, 5]]
            )
            predicted = coarse_scores.argmax(1)
            metrics = metrics_from_raw(true_coarse, predicted, 3)
            historical = np.load(prediction_path, allow_pickle=False)
            historical_sha_ok = sha256_file(prediction_path) == expected["prediction_sha256"]
            label_parity = np.array_equal(result["labels"], historical["labels"])
            score_difference = float(
                np.max(np.abs(result["probabilities"] - historical["probabilities"]))
            )
            family_results[subject] = {
                **metrics,
                "inference_seconds": elapsed,
                "checkpoint_sha256": checkpoint_sha,
                "feature_cache_sha256": sha256_file(cache),
                "historical_feature_provenance_sha_verified": True,
                "historical_extraction_manifest_sha256": sha256_file(extraction_path),
                "backbone_sha256_from_historical_manifest": cache_evidence["weights_sha256"],
                "sample_period_median_seconds": float(np.median(np.diff(embeddings.timestamps))),
                "historical_prediction_sha_ok": historical_sha_ok,
                "historical_label_parity": label_parity,
                "historical_score_max_abs_difference": score_difference,
                "probability_parity_at_1e_5": score_difference <= 1e-5,
            }
            np.savez_compressed(
                private / f"{name}_actual_forward.npz",
                labels=true_coarse,
                predictions=predicted,
                scores=coarse_scores,
                centers=result["centers"][valid],
            )
            diagnostics["checkpoints"].append(
                {
                    "path": str(checkpoint.relative_to(ROOT)),
                    "sha256": checkpoint_sha,
                    "training_subject_ids": train,
                    "intended_heldout_subject": subject,
                    "validation_subject_ids": [],
                    "validation_evidence": "fold trainer contains no validation/model selection loop",
                    "architecture": type(model).__name__,
                    "gru_input_shape": [None, 9, 512],
                    "state_dict_strict_load": True,
                    "actual_forward_completed": True,
                    "historical_summary": str(summary_path.relative_to(ROOT)),
                    "historical_summary_sha256": sha256_file(summary_path),
                    "feature_contract_from_source": FEATURE_CONTRACT,
                    "online_status": "CHECKPOINT_PROVENANCE_NOT_PROVABLE",
                    "online_blockers": [
                        "NO_COMPLETE_RUNTIME_MANIFEST",
                        "CENTERED_TRAINING_NOT_VALIDATED_FOR_CAUSAL_GATING",
                    ],
                    "training_dataset_provenance": "RECONSTRUCTED_HISTORICAL_DEVELOPMENT; not a data-rights approval",
                }
            )
            all_labels.append(true_coarse)
            all_predictions.append(predicted)
        pooled = metrics_from_raw(np.concatenate(all_labels), np.concatenate(all_predictions), 3)
        if not np.array_equal(
            np.sum([v["confusion_matrix"] for v in family_results.values()], axis=0),
            pooled["confusion_matrix"],
        ):
            raise ValueError("PER_SUBJECT_AGGREGATION_MISMATCH")
        diagnostics["families"][family] = {
            "per_subject": family_results,
            "pooled": pooled,
            "worst_subject_phone_recall": min(v["recall"][1] for v in family_results.values()),
        }
    write_json(output / "checkpoint_provenance.json", diagnostics)
    return diagnostics


def multiview_metadata(cohort: dict, output: Path) -> dict:
    records = []
    for session in cohort["sessions"]:
        parent = Path(session["video_path"]).parent
        streams = []
        for view in ("body", "face", "hands"):
            path = Path(session["video_path"].replace("_body.mp4", f"_{view}.mp4"))
            if not path.is_file():
                streams.append({"view": view.upper(), "status": "MISSING_LOCAL_FILE"})
                continue
            cap = cv2.VideoCapture(str(path))
            streams.append(
                {
                    "view": view.upper(),
                    "path": str(path.relative_to(ROOT)),
                    "sha256": sha256_file(path),
                    "fps": cap.get(cv2.CAP_PROP_FPS),
                    "frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
                    "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                    "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                    "status": "LOCAL_CONTAINER_READABLE" if cap.isOpened() else "UNREADABLE",
                }
            )
            cap.release()
        records.append(
            {
                "subject_id": session["subject_id"],
                "streams": streams,
                "synchronization": "NOT_CONTENT_VALIDATED; matching containers do not prove zero offset",
                "camera_orientation": "NOT_VERIFIED",
                "seat_calibration": "MISSING",
                "directory": str(parent.relative_to(ROOT)),
            }
        )
    result = {
        "subjects": records,
        "full_cohort_fusion_ready": False,
        "association_metrics_status": "ROI_ASSOCIATION_METRICS_NOT_REPRODUCIBLE",
        "source_audit": "build_v21_metadata.py hardcodes containment/exclusion percentages; no raw validation records",
    }
    write_json(output / "multiview_feasibility.json", result)
    return result


def fp_fn_audit(cohort: dict, output: Path) -> dict:
    baseline_path = ROOT / "reports/v20_ai85_execution/v20_raw_diagnostic_ledger.json"
    baseline = read_json(baseline_path)
    matches = (
        read_json(output / "full_event_matches.json")
        if (output / "full_event_matches.json").is_file()
        else {}
    )
    cases = []
    for session in cohort["sessions"]:
        sid = session["subject_id"]
        ground_truth = [
            interval.to_dict()
            for interval in parse_dmd_openlabel(
                Path(session["annotation_path"]),
                fps=session["fps"],
                total_frames=session["source_frames"],
            ).phone_intervals
        ]
        seen_ground_truth = set()
        duplicate_predictions = []
        for match in baseline[sid]["matched_events"]:
            target = match["ground_truth"]
            identity = (target["start_seconds"], target["end_seconds"])
            if identity in seen_ground_truth:
                duplicate_predictions.append(match["prediction"])
            seen_ground_truth.add(identity)
        fp_events = baseline[sid]["unmatched_predictions"] + duplicate_predictions
        trace_path = output / "private" / sid / "trace.jsonl"
        trace = (
            [json.loads(line) for line in trace_path.read_text(encoding="utf-8").splitlines()]
            if trace_path.is_file()
            else []
        )
        for kind, key in (("FP", "unmatched_predictions"), ("FN", "unmatched_gt")):
            for index, event in enumerate(fp_events if kind == "FP" else baseline[sid][key]):
                start, end = event["start_seconds"], event["end_seconds"]
                inside = [r for r in trace if start <= r["observation"]["timestamp"] <= end]
                nearest = (
                    min(trace, key=lambda r: abs(r["observation"]["timestamp"] - start))
                    if trace
                    else None
                )
                current = matches.get("B0", {}).get(sid, {})
                current_unmatched = current.get(key, [])
                persists = any(
                    abs(e["start_seconds"] - start) < 0.05 and abs(e["end_seconds"] - end) < 0.05
                    for e in current_unmatched
                )
                cases.append(
                    {
                        "event_id": f"{kind}_{sid}_{index + 1:02d}",
                        "kind": kind,
                        "subject_id": sid,
                        "session_id": session["session_id"],
                        "start_seconds": start,
                        "end_seconds": end,
                        "historical_record": event,
                        "duplicate_prediction_counted_as_fp": kind == "FP"
                        and event in duplicate_predictions,
                        "gt_overlaps": [
                            {
                                "start_seconds": g["start_seconds"],
                                "end_seconds": g["end_seconds"],
                                "intersection_seconds": min(end, g["end_seconds"])
                                - max(start, g["start_seconds"]),
                            }
                            for g in ground_truth
                            if min(end, g["end_seconds"]) > max(start, g["start_seconds"])
                        ],
                        "supporting_artifact": str(trace_path.relative_to(ROOT)),
                        "sampled_frames_in_interval": len(inside),
                        "frames_with_phone_proposals": sum(bool(r["candidates"]) for r in inside),
                        "nearest_start_trace": nearest,
                        "same_boundary_unmatched_in_B0": persists,
                        "root_cause_status": "DUPLICATE_SCORING_VERIFIED"
                        if kind == "FP" and event in duplicate_predictions
                        else "UNVERIFIED_ROOT_CAUSE",
                        "root_cause_confidence": None,
                        "first_incorrect_stage": "TEMPORAL_DUPLICATE_OR_FRAGMENT_ALERT"
                        if kind == "FP" and event in duplicate_predictions
                        else "NOT_ESTABLISHED_FROM_AUTOMATIC_TRACE_ALONE",
                        "B1_B2_rejection_stage": "DRIVER_ASSOCIATION_UNVERIFIED"
                        if inside
                        else "NO_SAMPLED_FRAME_INSIDE_INTERVAL",
                        "recovery_strategy": "verify role/view evidence and inspect detector/pose traces; no blindspot assertion",
                    }
                )
    result = {
        "source": str(baseline_path.relative_to(ROOT)),
        "source_sha256": sha256_file(baseline_path),
        "fp_count": sum(c["kind"] == "FP" for c in cases),
        "fn_count": sum(c["kind"] == "FN" for c in cases),
        "cases": cases,
        "semantic_visual_review": "NOT_ESTABLISHED; V21 handwritten categories are not validated labels",
    }
    if result["fp_count"] != sum(subject["fp"] for subject in baseline.values()):
        raise ValueError("FP_LEDGER_MUST_INCLUDE_DUPLICATE_MATCH_PENALTIES")
    write_json(output / "fp_fn_ledger.json", result)
    return result


def extract_case_evidence(cohort: dict, ledger: dict, output: Path) -> None:
    """GT-informed diagnostic frames only; never consumed by the autonomous runner."""
    folder = output / "private/case_frames"
    folder.mkdir(parents=True, exist_ok=True)
    by_subject = {s["subject_id"]: s for s in cohort["sessions"]}
    thumbnails = []
    for case in ledger["cases"]:
        session = by_subject[case["subject_id"]]
        evidence = []
        for view in ("body", "face", "hands"):
            path = Path(session["video_path"].replace("_body.mp4", f"_{view}.mp4"))
            if not path.is_file():
                evidence.append({"view": view.upper(), "status": "MISSING_LOCAL_FILE"})
                continue
            cap = cv2.VideoCapture(str(path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            for position, timestamp in enumerate(
                (
                    case["start_seconds"],
                    (case["start_seconds"] + case["end_seconds"]) / 2,
                    case["end_seconds"],
                )
            ):
                index = min(total - 1, max(0, round(timestamp * fps)))
                cap.set(cv2.CAP_PROP_POS_FRAMES, index)
                ok, frame = cap.read()
                if not ok or frame is None:
                    evidence.append(
                        {"view": view.upper(), "frame_index": index, "status": "DECODE_FAILED"}
                    )
                    continue
                artifact = folder / f"{case['event_id']}_{view}_{position}.jpg"
                if not cv2.imwrite(str(artifact), frame):
                    raise OSError(f"Cannot write evidence frame {artifact}")
                evidence.append(
                    {
                        "view": view.upper(),
                        "frame_index": index,
                        "timestamp_seconds": index / fps,
                        "path": str(artifact.relative_to(ROOT)),
                        "sha256": sha256_file(artifact),
                        "scope": "GT_INFORMED_DIAGNOSTIC",
                        "cross_view_sync": "UNVERIFIED",
                    }
                )
                if view == "body" and position == 1:
                    tile = np.zeros((205, 320, 3), dtype=np.uint8)
                    tile[:180] = cv2.resize(frame, (320, 180), interpolation=cv2.INTER_AREA)
                    cv2.putText(
                        tile,
                        f"{case['event_id']} t={index / fps:.2f}",
                        (4, 197),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.4,
                        (255, 255, 255),
                        1,
                    )
                    thumbnails.append(tile)
            cap.release()
        case["visual_evidence"] = evidence
    for page in range(0, len(thumbnails), 6):
        tiles = thumbnails[page : page + 6]
        tiles.extend(np.zeros((205, 320, 3), dtype=np.uint8) for _ in range(6 - len(tiles)))
        sheet = np.vstack([np.hstack(tiles[:2]), np.hstack(tiles[2:4]), np.hstack(tiles[4:6])])
        cv2.imwrite(str(folder / f"body_contact_{page // 6 + 1}.jpg"), sheet)
    ledger["diagnostic_frame_count"] = sum(
        "sha256" in item for case in ledger["cases"] for item in case["visual_evidence"]
    )
    write_json(output / "fp_fn_ledger.json", ledger)


def seatbelt_audit(output: Path) -> dict:
    from training.evaluate_seatbelt_v6 import evaluate_file

    source = ROOT / "reports/calibration/seatbelt_classifier_raw_predictions.csv"
    result = evaluate_file(source, output / "seatbelt_metrics_recomputed.json")
    result["scope"] = (
        "RECOMPUTED_EXISTING_VALIDATION_PREDICTIONS; not new model inference or independent holdout"
    )
    result["exact_checkpoint_for_csv"] = "NOT_ESTABLISHED_FROM_CSV"
    result["worst_subject_metrics"] = "BLOCKED_MISSING_SUBJECT_IDENTIFIERS_IN_CSV"
    result["current_review_status"] = (
        "NO_LABEL_APPROVAL_PERFORMED; inherited pending-review count is not a live count"
    )
    result["fitting_performed"] = False
    # Independent sklearn reconciliation from the same real raw records.
    with source.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    classes = ["seatbelt_fastened", "seatbelt_unfastened", "uncertain_or_occluded"]
    labels = np.array([classes.index(row["true_class"]) for row in rows])
    predictions = np.array(
        [
            [float(row[k]) for k in ("prob_fastened", "prob_unfastened", "prob_uncertain")]
            for row in rows
        ]
    ).argmax(1)
    result["sklearn_reconciliation"] = metrics_from_raw(labels, predictions, 3)
    write_json(output / "seatbelt_metrics_recomputed.json", result)
    return result


def pipeline_graph(output: Path) -> dict:
    nodes = []
    declarations = [
        (
            "VIDEO",
            "training/dmd/run_autonomous_v20.py",
            "run_session_inference",
            "HWC uint8 BGR at native resolution",
            "cv2.VideoCapture; stride15; exact frame grid",
        ),
        (
            "DRIVER_ROI",
            "training/dmd/run_autonomous_v20.py",
            "run_session_inference",
            "HWC uint8 BGR crop",
            "static [0,.10,.65,.95]; not calibrated role evidence",
        ),
        (
            "OCCUPANT_POSE",
            "backend/ai/auxiliary.py",
            "PoseEstimator",
            "BGR crop, YOLO imgsz480; COCO17",
            "candidate frames only for exact B0 exposure; no role identity",
        ),
        (
            "PHONE_DETECTOR",
            "training/dmd/run_autonomous_v20.py",
            "run_session_inference",
            "BGR crop, YOLO imgsz640",
            "specialist plus auxiliary COCO class67, conf.15",
        ),
        (
            "PHONE_CANDIDATES",
            "training/dmd/run_autonomous_v20.py",
            "merge_detections",
            "local/global xyxy float",
            "cross-source NMS IoU.50; all proposals saved",
        ),
        (
            "DRIVER_ASSOCIATION",
            "training/dmd/driver_phone_association.py",
            "DriverPhoneAssociation",
            "phone Box + verified OccupantTrack tuple",
            "B1/B2 UNKNOWN without calibrated role; no nearest-person driver assignment",
        ),
        (
            "ACTION",
            "training/dmd/action_adapter.py",
            "ActionInferenceAdapter",
            "9x512 embeddings or heuristic pose features",
            "B2 RULE_ONLY; online V6/V7 blocked; diagnostics separately centered",
        ),
        (
            "FUSION",
            "training/dmd/run_verified_v22.py",
            "replay_verified_record",
            "raw candidates/pose without GT",
            "role fail-closed; no claimed measured action benefit",
        ),
        (
            "TEMPORAL",
            "backend/ai/events.py",
            "TemporalEventEngine",
            "timestamped Observation",
            "activation alert window, cooldown, no finalized interval tracking",
        ),
        (
            "PREDICTED_EVENTS",
            "training/dmd/run_verified_v22.py",
            "deduplicate",
            "activation intervals",
            "original cooldown merge, never extends to GT boundaries",
        ),
        (
            "FROZEN_MATCHER",
            "training/dmd/temporal_eval.py",
            "match_temporal_events",
            "events + Stage2-only OpenLABEL intervals",
            "IoU.3 OR onset<=2.5/inter>.5 OR containment>=.4/inter>=.8; offset tolerance unused",
        ),
    ]
    for stage, path, function, shape, scope in declarations:
        nodes.append(
            {
                "stage": stage,
                "file": path,
                "function": function,
                "file_sha256": sha256_file(ROOT / path),
                "feature_input_shape": shape,
                "inference_scope": scope,
                "runtime": "NOT_SEPARATELY_INSTRUMENTED; session wall time measured",
                "error_handling": "missing/checksum/decode errors abort; no subject skipping",
                "provenance": "direct source inspection",
            }
        )
    graph = {
        "nodes": nodes,
        "edges": [[a[0], b[0]] for a, b in zip(declarations, declarations[1:], strict=False)],
        "backend_entrypoint": "InferenceWorker.run -> VideoAnalyzer.analyze -> _analyze_context -> SafetyDetector/PoseEstimator/OccupantAssociator -> TemporalEventEngine",
        "backend_action_integration": "ActionInferenceAdapter is research-runner-only; VideoAnalyzer does not import it",
        "source_imports": {
            path: sorted(
                {
                    node.module
                    for node in ast.walk(ast.parse((ROOT / path).read_text(encoding="utf-8")))
                    if isinstance(node, ast.ImportFrom) and node.module
                }
            )
            for _, path, *_ in declarations
        },
    }
    write_json(output / "pipeline_graph.json", graph)
    return graph


def test_summary(output: Path) -> dict:
    summaries = {}
    for name in ("targeted", "full"):
        path = output / "logs" / f"{name}.xml"
        if not path.is_file():
            summaries[name] = {"status": "NOT_RUN_OR_NO_JUNIT_ARTIFACT"}
            continue
        root = ET.parse(path).getroot()
        suites = list(root.iter("testsuite"))
        counts = {
            key: sum(int(s.get(key, 0)) for s in suites)
            for key in ("tests", "failures", "errors", "skipped")
        }
        summaries[name] = {
            **counts,
            "tests": len(list(root.iter("testcase"))),
            "subtests": counts["tests"] - len(list(root.iter("testcase"))),
            "junit_reported_total_including_subtests": counts["tests"],
            "passed": len(list(root.iter("testcase")))
            - counts["failures"]
            - counts["errors"]
            - counts["skipped"],
            "source_sha256": sha256_file(path),
        }
    checks = output / "logs/check_results.json"
    summaries["checks"] = read_json(checks) if checks.is_file() else {"status": "NOT_RUN"}
    write_json(output / "tests_summary.json", summaries)
    return summaries


def markdown(output: Path, name: str, title: str, paragraphs: list[str]) -> None:
    (output / name).write_text(
        "# " + title + "\n\n" + "\n\n".join(paragraphs) + "\n", encoding="utf-8"
    )


def reports(
    output: Path,
    claim: dict,
    diagnostics: dict,
    multiview: dict,
    ledger: dict,
    seatbelt: dict,
    tests: dict,
) -> None:
    results = read_json(output / "full_experiments.json")
    for run_id, result in results.items():
        blocked = result["status"] == "BLOCKED"
        result["actual_mode"] = (
            "NOT_RUN_BLOCKED" if blocked else ("RULE_ONLY" if run_id == "B2" else "NONE")
        )
        result["model_loaded"] = False  # action head only; detector hashes recorded separately
        result["calibration_status"] = "NOT_EVALUATED" if blocked else "UNCALIBRATED_SCORES"
        result["fallback_reason"] = result["blockers"] if blocked else None
        if blocked:
            write_json(output / f"full_predictions_{run_id}.status.json", result)
    write_json(output / "full_experiments.json", results)
    write_json(output / "full_metrics_comparison.json", results)
    historical = read_json(ROOT / "reports/v20_ai85_execution/event_matches.json")
    current_v21 = read_json(ROOT / "reports/v21_ai85_recovery/metrics_comparison.json")
    boundary_path = output / "boundary_experiment_summary.json"
    boundary = read_json(boundary_path) if boundary_path.is_file() else None
    manifest_path = output / "full_experiment_manifest.json"
    original_plan = output / "full_experiment_plan.json"
    if not original_plan.exists():
        original_plan.write_bytes(manifest_path.read_bytes())
    write_json(
        manifest_path,
        {
            "status": "COMPLETE_WITH_BLOCKED_RUNS",
            "original_plan_sha256": sha256_file(original_plan),
            "protocol_sha256": sha256_file(output / "evaluation_protocol.json"),
            "runs": {
                run_id: {key: value for key, value in run.items() if key != "sessions"}
                for run_id, run in results.items()
            },
            "separate_boundary_replay": boundary,
        },
    )
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    runtime = {
        "hardware": {
            "platform": platform.platform(),
            "processor": platform.processor(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "device_used": "CPU",
            "torch_threads_for_head_diagnostics": 1,
        },
        "full_session_shared_inference": [],
        "run_timings": {k: v.get("replay_and_scoring_wall_seconds") for k, v in results.items()},
        "GPU_benchmark": "BLOCKED_NO_CUDA_TORCH",
        "interpretation": "shared extraction time counted once; replay is not full-model FPS",
        "boundary_replay": boundary,
    }
    for session in read_json(output / "cohort_manifest.json")["sessions"]:
        path = output / "private" / session["subject_id"] / "inference_summary.json"
        if path.is_file():
            measured = read_json(path)
            runtime["full_session_shared_inference"].append(
                {
                    "subject_id": session["subject_id"],
                    "processed_frames": measured["processed_frames"],
                    "wall_seconds": measured["wall_time_seconds"],
                    "sampled_fps": measured["processed_frames"] / measured["wall_time_seconds"],
                }
            )
    write_json(output / "runtime_benchmark.json", runtime)
    decision = {
        "git_head": head,
        "engineering_status": "PARTIAL_P0_FIXED_FULL_DEEP_COMPARISON_BLOCKED",
        "scientific_improvement": "VERIFIED_DRIVER_IMPROVEMENT_NOT_ESTABLISHED",
        "legacy_role_temporal_boundary_experiment": boundary,
        "ai85_status": "AI85_TARGET_NOT_MET",
        "independent_status": "INDEPENDENT_AI85_NOT_ESTABLISHED",
        "production_approved": False,
        "training_performed": False,
        "protected_subjects_accessed": [],
        "user_prompt_modified_by_agent": False,
        "blockers": [
            "NO_VERIFIED_SEAT_CALIBRATION_OR_OCCUPANT_ROLE_GROUND_TRUTH",
            "CHECKPOINT_PROVENANCE_NOT_PROVABLE_FOR_ONLINE_GATING",
            "CENTERED_FEATURE_LABEL_TIMING_NOT_VALIDATED_FOR_CAUSAL_INFERENCE",
            "ACTION_METRICS_NOT_REPRODUCIBLE_FOR_CLAIMED_3600_HYBRID_SAMPLES",
            "NO_INDEPENDENT_UNTOUCHED_HOLDOUT",
            "SEATBELT_LABEL_REVIEW_AND_RIGHTS_REQUIRED_BEFORE_FITTING",
        ],
        "next_actions": [
            "collect verified view/seat/occupant association labels with camera orientation",
            "reconstruct complete checkpoint/feature/time contracts and validate causal alignment on legitimate development splits",
            "human review seatbelt uncertain labels and lock subject-disjoint splits/rights before any warm-start",
        ],
    }
    write_json(output / "candidate_readiness.json", decision)
    markdown(
        output,
        "01_V21_CLAIM_VS_EVIDENCE_AUDIT.md",
        "V21 claim versus evidence",
        [
            f"Audited local and remote HEAD: `{head}`. Newer than prompt's 494ec429. User prompt retained. Source history proves adapter and runner were introduced in 494ec429; auxiliary change in that commit only normalizes Path input.",
            f"Historical V20 scope: HISTORICAL_FULL_COHORT_BASELINE, TP={historical['true_positives']} FP={historical['false_positives']} FN={historical['false_negatives']} on {historical['evaluated_minutes']} min / {historical['total_gt_events']} GT. This is inherited evidence, not a fresh run.",
            "Current V21 artifacts were updated by 98502a4 after the prompt baseline. They now report full-cohort exposure, so the prompt's earlier one-event description is historical. Recomputed file coverage and current fresh B0 are evaluated separately; V21 metrics remain REPORTED_NOT_YET_FULLY_REPRODUCED.",
            "See claim_audit.json for current table, counts/hashes and score constancy. No original 3600 hybrid truth-label records were located. ROI percentages are literals in build_v21_metadata.py, not computed validation metrics.",
            "Source graph: pipeline_graph.json. Production VideoAnalyzer does not run this action adapter. A research benchmark does not establish production behavior.",
        ],
    )
    write_json(
        output / "claim_audit.json",
        {
            "git_head": head,
            "historical_v20": historical,
            "current_v21_table": [
                {k: v for k, v in row.items() if k != "per_subject"} for row in current_v21["table"]
            ],
            "action": claim,
        },
    )
    markdown(
        output,
        "02_LOSO_CHECKPOINT_SAFETY_FIX.md",
        "LOSO checkpoint safety",
        [
            "Removed both V6 and V7 other-fold fallbacks. Exact subject path is deterministic. Require matching SHA, split, architecture, preprocessing, class mapping, dataset provenance and backbone identity before online activation. ERROR mode fails closed; explicit RULE_ONLY mode records requested/actual mode, missing manifest reason and model_loaded=false.",
            "Regression failing_before.log demonstrates deserialization of the wrong fold before the fix. Targeted/full test logs contain current outcomes. A matching checkpoint filename is insufficient proof.",
            "Six original heads were loaded strictly and executed offline against development feature caches. checkpoint_provenance.json records real checkpoint hashes and split metadata. These diagnostics do not create an online compatibility manifest or retroactively approve data rights.",
        ],
    )
    markdown(
        output,
        "03_ACTION_PREPROCESSING_AND_CLASS_MAPPING_AUDIT.md",
        "Action inputs and output semantics",
        [
            "V6 coarse_targets maps flat 0 to SAFE, flat 1..4 to PHONE and flat 5 to OTHER. Hierarchical coarse softmax is the correct SAFE/PHONE/OTHER score. Fine phone head only subdivides PHONE. V7 multiplies binary NON_PHONE/PHONE gate with conditional SAFE/OTHER or four phone classes. Regression reconstructs coarse scores from six-class output.",
            "Training: native static ROI -> cv2 AREA 64 RGB -> torch bilinear 112, align_corners=false -> ImageNet normalization -> frozen ImageNet ResNet18 -> cached float16 embeddings -> float32 [B,9,512]. Center-frame labels, complete windows, no padding. Original adapter directly resized112 and repeated first embeddings; both differed.",
            "Adapter extraction now matches spatial/precision preprocessing, history is bounded, resets per subject/session/camera/track or timestamp gap, and refuses insufficient history. Centered labels cannot automatically serve causal current-frame gating; online deep activation remains blocked. Rule pose features include heuristic elbow/shoulder estimates and filtered keypoint identity limitations; scores are priors, not calibrated probabilities.",
            "Actual offline forwards and archival score parity are in checkpoint_provenance.json. These use already-extracted features, so they do not benchmark full visual backbone FPS or validate current online preprocessing.",
        ],
    )
    markdown(
        output,
        "04_CONFUSION_MATRIX_RECONCILIATION.md",
        "Action confusion matrix reconciliation",
        [
            f"Published matrix arithmetic Macro-F1 = {claim['matrix_arithmetic']['macro_f1']:.12f} ({100 * claim['matrix_arithmetic']['macro_f1']:.6f}%). Reported85.16% is inconsistent. Row support sums to {claim['matrix_arithmetic']['sample_count']}, which proves arithmetic only.",
            f"V21 action trace contains {claim['raw_prediction_rows']} rows, {claim['rows_with_ground_truth']} with GT labels and {claim['distinct_score_triplets']} distinct score triplets. Status ACTION_METRICS_NOT_REPRODUCIBLE for the claimed hybrid experiment. No raw labels were manufactured from the matrix.",
            "Independent sklearn recomputation is performed on real cached-feature checkpoint forward predictions for V6/V7, and on real seatbelt CSV labels. These are distinct experiments and cannot validate the published hybrid claim.",
        ],
    )
    table = [
        "| Run | Status | Minutes | GT | TP/FP/FN | Precision | Recall | F1 | FA/min |",
        "|---|---|---:|---:|---|---:|---:|---:|---:|",
    ]
    for key, value in results.items():
        if "tp" in value:
            table.append(
                f"| {key} | {value['status']} | {value['evaluated_minutes']:.6f} | {value['gt_count']} | {value['tp']}/{value['fp']}/{value['fn']} | {value['precision']:.6f} | {value['recall']:.6f} | {value['f1']:.6f} | {value['fa_per_minute']:.6f} |"
            )
        else:
            table.append(f"| {key} | BLOCKED | — | — | — | — | — | — | — |")
    if boundary:
        table.append(
            f"| T1 separate legacy-role boundary replay | COMPLETE | {boundary['evaluated_minutes']:.6f} | {boundary['gt_count']} | {boundary['tp']}/{boundary['fp']}/{boundary['fn']} | {boundary['precision']:.6f} | {boundary['recall']:.6f} | {boundary['f1']:.6f} | {boundary['fa_per_minute']:.6f} |"
        )
    markdown(
        output,
        "05_FULL_COHORT_EXPERIMENTS.md",
        "Full cohort controlled experiments",
        [
            "\n".join(table),
            "B0 is exact V20 D5 inference with full persistence and Stage2 separated. Replay parity checked on activation intervals. B1/B2 use the same actual detections/pose exposure and conservative UNKNOWN roles because no verified calibration was found. Association coverage is zero; losing all true events is a failure, not an improvement. B1 also changes proximity geometry; no isolated association-effect claim is made.",
            "B3/B4 did not evaluate any frames or GT events. Empty prediction files have BLOCKED status sidecars and contain no negative evidence. No smoke result is used as a full result. Protocol, model/config/matcher/prediction hashes are persisted. Resume reuses completed sessions only; interrupted attempts remain intact.",
        ],
    )
    markdown(
        output,
        "06_16_FP_14_FN_EVIDENCE_AUDIT.md",
        "Original FP/FN evidence ledger",
        [
            f"Rebuilt {ledger['fp_count']} historical FP and {ledger['fn_count']} historical FN cases from V20 raw diagnostic ledger, including duplicate-match FP penalties. Extracted {ledger.get('diagnostic_frame_count', 0)} diagnostic frames from locally available views. Each case has source interval, session, nearest fresh trace, GT overlap, candidate availability and current matching status. fp_fn_ledger.json is local and ignored by Git.",
            "Root causes remain UNVERIFIED_ROOT_CAUSE where no validated visual/role evidence establishes the first wrong stage. Original V21 console/steering/AC/blindspot narratives are not accepted as ground truth. Zero false alerts through universal role abstention does not demonstrate recovery; B1/B2 lose retained true events too.",
            "All available BODY/FACE/HANDS views are inventoried separately. Absent FACE/HANDS for one subject prevents full-cohort view conclusions. No DASHBOARD stream is assumed.",
        ],
    )
    markdown(
        output,
        "07_DRIVER_ASSOCIATION_AND_MULTIVIEW.md",
        "Driver association and multiview",
        [
            "Fixed pose-present and pose-missing paths that could create driver identity without role evidence. Highest-confidence pose in a crop is now unverified UNKNOWN in both V21 entrypoints. Passenger and verified-mounted rejection tests remain applicable; unverified mounted suggestions remain UNKNOWN.",
            "V21 enable_dynamic_roi was unused; actual crop is static. No calibrated tracking or real association ground-truth set was found. Driver association precision, passenger exclusion, true-driver retention and 99.6% containment cannot be claimed. ROI_ASSOCIATION_METRICS_NOT_REPRODUCIBLE.",
            "multiview_feasibility.json records actual file hashes, FPS and frame counts. gZ-36 has BODY/FACE/HANDS locally; gC-14 has BODY only. Content synchronization, offsets and orientation are not proven by file presence. Full-cohort fusion is blocked pending verified synchronization/calibration and missing views.",
        ],
    )
    markdown(
        output,
        "08_TEMPORAL_AND_SMALL_PHONE_ABLATIONS.md",
        "Temporal boundaries and detector feasibility",
        [
            "Frozen matcher bytes and policy bytes are unchanged. Matcher can attach multiple predictions to one GT; scoring counts duplicates as FP. This is not a strict one-to-one implementation, although unique-GT TP accounting is preserved. offset_tolerance_seconds is not used by matching. Published V21 0.05/4s policy does not match actual frozen .3/2.5s policy.",
            "full_event_matches.json reports IoU, onset/offset errors, actual prediction/GT durations, union coverage and fragments separately. Temporal engine emits activation windows once; it does not finalize complete event boundaries on release or EOF. No synthetic EOF tail is added. A short alert is not a fully tracked GT event.",
            "Official baseline uses stride15. Stride10/8/adaptive and SAHI AP ablations have not been run; no fully reviewed exhaustive original-frame detection annotation set was established. Additional phone proposals alone are not AP/recall evidence. Multiview fusion also remains blocked by sync/role evidence. These are explicit unperformed experiments, not measured improvements.",
            (
                f"Separate T1 boundary replay: TP={boundary['tp']}, FP={boundary['fp']}, FN={boundary['fn']}; "
                f"Precision={boundary['precision']:.6f}, Recall={boundary['recall']:.6f}, F1={boundary['f1']:.6f}, "
                f"FA/min={boundary['fa_per_minute']:.6f} on {boundary['evaluated_minutes']:.6f} minutes / {boundary['gt_count']} GT. "
                "Reuses exact hashed B0 autonomous observations and the unchanged event engine/matcher; only interval reporting changes. "
                "Ends advance only to physically positive observations while the engine is active. EOF does not invent a tail. "
                "This is exploratory development evidence with legacy B0 driver-role assumptions; it does not establish verified association or independent AI85. "
                "Protocol and summary are boundary_experiment_protocol.json / boundary_experiment_summary.json; detailed matches stay under private/."
                if boundary
                else "Temporal boundary replay: NOT_RUN."
            ),
            (
                f"Retention check: recovered {boundary.get('recovered_old_fn_count', 'NOT_RECONCILED')} original FN; "
                f"{boundary.get('old_fn_remaining_count', 'NOT_RECONCILED')} original FN remain; "
                f"lost {boundary.get('lost_old_tp_count', 'NOT_RECONCILED')} original TP. "
                "FP reduction is under the frozen event accounting, not proof of rejecting nine physically false phone boxes. "
                "Changed boundaries can alter greedy event assignment; inspect the recovered/lost event IDs before adoption."
                if boundary
                else "No boundary retention result."
            ),
        ],
    )
    markdown(
        output,
        "09_SEATBELT_RECOVERY.md",
        "Seatbelt recovery audit",
        [
            f"Recomputed existing {seatbelt['sample_count']}-row validation CSV with sklearn reconciliation. Raw Macro-F1={seatbelt['raw']['macro_f1']:.6f}; post-policy Macro-F1={seatbelt['post_policy']['macro_f1']:.6f}; UNCERTAIN recall={seatbelt['raw']['per_class']['uncertain_or_occluded']['recall']:.6f}. This is cached validation prediction evidence, not new model inference or independent performance.",
            "Different historical 58.52% result uses a different policy/TTA variant; do not choose the higher score as a new baseline. Current CSV lacks subject IDs and exact checkpoint linkage, so worst-subject metrics and full provenance remain blocked. seatbelt_metrics_recomputed.json includes per-class F1, false-unfastened rate and abstention.",
            "No pending labels were approved, no rights flags changed and no warm-start training performed. Need human-reviewed difficult uncertain/occluded torso cases, identity/session groups, clear rights and a locked subject split before fitting.",
        ],
    )
    markdown(
        output,
        "10_RUNTIME_AND_TESTS.md",
        "Runtime and validation",
        [
            "runtime_benchmark.json contains actual CPU session extraction timings and separate cached-head/replay timings. Torch is CPU-only; no GPU speed is reported. Shared detector extraction is reused by B0/B1/B2, not multiplied into independent runtime claims.",
            "tests_summary.json is parsed from current JUnit XML and recorded native exit codes. logs preserve failing-before and passing-after evidence. No inherited 667-pass count is treated as current verification.",
            "Engineering remains partial: unsafe fallbacks, coverage loss and role fabrication paths are fixed, but full deep/action/association scientific validation is blocked. No production model was activated.",
        ],
    )
    markdown(
        output,
        "12_FINAL_AI_LEAD_DECISION.md",
        "Final AI lead decision",
        [
            f"HEAD `{head}`. AI85_TARGET_NOT_MET. INDEPENDENT_AI85_NOT_ESTABLISHED. production_approved=false. Verified-driver candidate improvement remains unestablished.",
            (
                f"A separate controlled boundary replay on the same full legacy-role observations improves event F1 from {historical['f1']:.4f} to {boundary['f1']:.4f}: "
                f"TP/FP/FN={boundary['tp']}/{boundary['fp']}/{boundary['fn']}. "
                "This is a real development replay result, not B3/B4, not an action-model gain, and not independent/verified-driver validation."
                if boundary
                else "Boundary replay not completed."
            ),
            "Full B0/B1/B2 comparisons are available with raw predictions; B3/B4 remain blocked by online checkpoint provenance and centered temporal compatibility. Offline V6/V7 cached-feature LOSO forwards are measured separately and are not hybrid/online validation.",
            "Next: (1) verify camera/seat/occupant identity and collect role/phone association negatives, (2) complete checkpoint and causal feature/time provenance before gating, (3) human-review seatbelt difficult labels and lock rights/splits before warm-start. Do not retrain from scratch to hide inference/evaluation correctness problems.",
            "gA-1 and protected gE-28/gZ-37 data were not opened or fitted. No private media/labels uploaded. No commit/push/merge, checkpoint deletion, dataset deletion, data-rights approval or production activation performed.",
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()
    output = args.output
    claim = action_claim_audit(output)
    diagnostics = checkpoint_diagnostics(output)
    cohort = read_json(output / "cohort_manifest.json")
    multiview = multiview_metadata(cohort, output)
    ledger = fp_fn_audit(cohort, output)
    extract_case_evidence(cohort, ledger, output)
    seatbelt = seatbelt_audit(output)
    pipeline_graph(output)
    tests = test_summary(output)
    reports(output, claim, diagnostics, multiview, ledger, seatbelt, tests)
    print(
        json.dumps(
            {
                "action_matrix_macro_f1": claim["matrix_arithmetic"]["macro_f1"],
                "offline_action": {
                    key: value["pooled"]["macro_f1"]
                    for key, value in diagnostics["families"].items()
                },
                "seatbelt_cached_validation_macro_f1": seatbelt["raw"]["macro_f1"],
                "tests": tests,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
