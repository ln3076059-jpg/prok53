"""V13 read-only diagnostics and evidence rendering."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import cv2
import numpy as np

from training.common import sha256_file
from training.dmd.behavior_roi import BehaviorRoiExtractor, Box, RoiStrategy
from training.dmd.high_res_roi import HighResolutionRoiPipeline, frame_sha256
from training.dmd.research_permissions import read_permissions, require_operation, require_subject

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v13_research"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def high_resolution_audit() -> dict:
    require_operation("diagnostics")
    source_cases = load("reports/v12_research/roi_audit.json")["cases"]
    fn_cases = {
        row["event_id"]: row
        for row in load("reports/v11_research/fn_root_cause.json")["cases"]
    }
    captures: dict[str, cv2.VideoCapture] = {}
    rows, timings = [], []
    try:
        for source in source_cases:
            require_subject(source["subject"])
            case = fn_cases[source["case_id"]]
            path = ROOT / source["original_video_path"]
            if source["subject"] not in captures:
                if sha256_file(path) != source["original_video_sha256"]:
                    raise ValueError("source video hash changed")
                captures[source["subject"]] = cv2.VideoCapture(str(path))
            capture = captures[source["subject"]]
            capture.set(cv2.CAP_PROP_POS_FRAMES, source["frame_index"])
            ok, bgr = capture.read()
            if not ok:
                raise ValueError(f"cannot decode {source['case_id']}")
            frame = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            driver_coords = case["object_detection"]["driver_roi_in_original_frame"]
            driver_source = "V11_FN_ROOT_CAUSE"
            if driver_coords is None:
                driver_coords = source["provenance"]["original_frame_box"]
                driver_source = "V12_VERIFIED_ROI_PROVENANCE_FALLBACK"
            driver = Box(*driver_coords)
            coords = case["object_detection"]["selected_box_in_original_frame"]
            phone = Box(*coords) if coords else None
            begin = time.perf_counter()
            baseline = BehaviorRoiExtractor().extract(
                frame, driver, RoiStrategy.FULL_DRIVER, source_camera="BODY"
            )[0]
            pipeline = HighResolutionRoiPipeline()
            variants = {
                "R1_ORIGINAL_DRIVER": pipeline.extract(
                    frame, driver, RoiStrategy.FULL_DRIVER, source_camera="BODY", phone_box=phone
                ),
                "R2_PADDED_HAND_PHONE": pipeline.extract(
                    frame, driver, RoiStrategy.HAND_PHONE, source_camera="BODY", phone_box=phone
                ),
                "R3_FACE_HAND": pipeline.extract(
                    frame, driver, RoiStrategy.FACE_HAND, source_camera="BODY"
                ),
                "R4_MULTI_REGION": pipeline.extract(
                    frame, driver, RoiStrategy.MULTI_REGION, source_camera="BODY", phone_box=phone
                ),
                "R5_SCALE_AWARE_DYNAMIC": pipeline.extract(
                    frame, driver, RoiStrategy.DYNAMIC_PADDED, source_camera="BODY", phone_box=phone
                ),
            }
            timings.append((time.perf_counter() - begin) * 1000)
            records = {key: [crop.record() for crop in crops] for key, crops in variants.items()}
            rows.append(
                {
                    "case_id": source["case_id"],
                    "subject": source["subject"],
                    "frame_index": source["frame_index"],
                    "source_video": source["original_video_path"],
                    "source_video_sha256": source["original_video_sha256"],
                    "decoded_frame_sha256": frame_sha256(frame),
                    "original_dimensions": [frame.shape[1], frame.shape[0]],
                    "phone_box": coords,
                    "driver_box_source": driver_source,
                    "phone_size": [phone.width, phone.height] if phone else None,
                    "R0_FROZEN_64": {
                        "output_dimensions": [baseline.image.shape[1], baseline.image.shape[0]],
                        "provenance": baseline.provenance.as_dict(),
                    },
                    "variants": records,
                    "output_bytes": int(
                        sum(c.image.nbytes for crops in variants.values() for c in crops)
                    ),
                    "scope": (
                        "oracle reviewed-box diagnostic; no detector AP or recovered-event claim"
                    ),
                }
            )
    finally:
        for capture in captures.values():
            capture.release()
    coverage = [
        crop["phone_coverage"]
        for row in rows
        for crop in row["variants"]["R5_SCALE_AWARE_DYNAMIC"]
        if crop["phone_coverage"] is not None
    ]
    return {
        "cases": rows,
        "decoded_frames": len(rows),
        "reviewed_boxes": len(coverage),
        "r5_full_phone_coverage": sum(value == 1.0 for value in coverage),
        "latency_ms": {str(p): float(np.percentile(timings, p)) for p in (50, 95, 99)},
        "detector_rerun": False,
        "phone_ap": None,
        "event_recovery_claimed": False,
    }


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    disk = shutil.disk_usage(ROOT)
    permissions = read_permissions()
    rights = {key: permissions.get(key) for key in (
        "RIGHTS_STATUS", "RIGHTS_ACCEPTED", "CANONICAL_ELIGIBLE",
        "TEMPORAL_DEVELOPMENT_ELIGIBLE", "UNTOUCHED_HOLDOUT_ELIGIBLE"
    )}
    environment = {
        "generated_at": datetime.now(UTC).isoformat(),
        "head": head,
        "branch": subprocess.check_output(
            ["git", "branch", "--show-current"], cwd=ROOT, text=True
        ).strip(),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "disk_total_bytes": disk.total,
        "disk_free_bytes": disk.free,
        "gpu": "NOT_AVAILABLE_NVIDIA_SMI",
    }
    write("environment.json", environment)
    write("high_resolution_roi.json", high_resolution_audit())
    dataset = load("reports/v12_research/dataset_manifest.json")
    dataset.update(
        audited_at=environment["generated_at"],
        audit_head=head,
        permissions=rights,
        permission_sha256=sha256_file(ROOT / "dataset/external_dmd/README.md"),
        protected_media_or_labels_opened=False,
    )
    write("dataset_manifest.json", dataset)
    write("model_manifest.json", {
        "source": "reports/v12_research/model_manifest.json",
        "source_sha256": sha256_file(ROOT / "reports/v12_research/model_manifest.json"),
        "inventory": load("reports/v12_research/model_manifest.json"),
        "fitted_e3_head_scaler": None,
        "checkpoint_activated": False,
    })
    metrics = load("reports/v12_research/metric_comparison.json")
    metrics.update(v13_new_accuracy_metrics=None, phone_ap=None, independent_ai85_established=False)
    write("metrics_comparison.json", metrics)
    write("event_match_records.json", {
        "source": "reports/v12_research/event_match_records.json",
        "source_sha256": sha256_file(ROOT / "reports/v12_research/event_match_records.json"),
        "records": load("reports/v12_research/event_match_records.json"),
        "v13_candidate_replay": "BLOCKED_MISSING_FITTED_HEAD_SCALER_AND_VERIFIED_OCCUPANTS",
        "matcher_changed": False,
    })
    seat = dataset["seatbelt_review"]
    readiness = {
        "online_e3_features": "READY_ENGINEERING_ONLY",
        "fitted_e3_head_scaler": "BLOCKED_NOT_AVAILABLE",
        "original_resolution_roi": "READY_DIAGNOSTIC_ONLY",
        "driver_association": "BLOCKED_NO_VERIFIED_OCCUPANT_GROUND_TRUTH",
        "event_replay": "BLOCKED_MISSING_FITTED_HEAD_SCALER_AND_VERIFIED_OCCUPANTS",
        "seatbelt_training": f"BLOCKED_{seat['status_counts'].get('PENDING', 0)}_PENDING_LABELS",
        "independent_evaluation": "BLOCKED_NO_NEW_LOCKED_HOLDOUT",
        "production": "REJECT",
    }
    write("candidate_readiness.json", readiness)
    write("experiment_registry.json", {
        "head": head,
        "level": "LEVEL_1_DIAGNOSTICS",
        "training_performed": False,
        "experiments": {
            "V13-R0-R5": {
                "decision": "KEEP_DIAGNOSTIC_PIPELINE",
                "artifact": "high_resolution_roi.json",
            },
            "V13-A0": {"decision": "KEEP_FROZEN_REFERENCE", "source": "V12"},
            "V13-A1-A6": {"decision": "BLOCKED", "reason": "DMD fitting rights denied"},
            "V13-EVENT": {"decision": "BLOCKED", "reason": readiness["event_replay"]},
            "V13-SEATBELT": {"decision": "BLOCKED", "reason": readiness["seatbelt_training"]},
        },
        "protected_holdouts_accessed": False,
        "production_approved": False,
    })


if __name__ == "__main__":
    run()
