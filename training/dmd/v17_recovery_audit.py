"""Generate V17 recovery artifacts from verified local evidence only."""

# ruff: noqa: E501

from __future__ import annotations

import json
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import yaml

from training.common import sha256_file, stable_json_hash
from training.dmd.research_permissions import read_permissions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v17_ai85_recovery"


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(name: str, payload: object) -> None:
    (OUT / name).write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    registry = read_json("reports/v16_data_ai85/dataset_registry.json")["registry"]
    inventory = read_json("reports/v16_data_ai85/data_manifest.json")
    metrics = read_json("reports/v16_data_ai85/metrics_comparison.json")
    model = read_json("reports/v16_data_ai85/model_manifest.json")
    permissions = read_permissions()

    blockers = {
        "DMD_2026": ["RIGHTS_TO_FIT_DENIED", "DOWNLOAD_AUTHORIZATION_DENIED", "NO_APPROVED_BBOX_BENCHMARK", "NO_NEW_INDEPENDENT_HOLDOUT"],
        "100_DRIVER": ["ACCESS_AGREEMENT_MISSING", "DOWNLOAD_AUTHORIZATION_DENIED", "HUMAN_LABEL_MAPPING_REVIEW_MISSING", "SPLIT_VERIFICATION_MISSING"],
        "STATE_FARM_DISTRACTED_DRIVER": ["COMPETITION_RULE_ACCEPTANCE_MISSING", "DOWNLOAD_AUTHORIZATION_DENIED", "SUBJECT_MANIFEST_UNAVAILABLE_BEFORE_ACCESS", "STATIC_IMAGES_HAVE_NO_EVENT_TIMESTAMPS"],
        "DRIVE_AND_ACT": ["PHONE_LABEL_EQUIVALENCE_UNVERIFIED", "DOWNLOAD_AUTHORIZATION_DENIED", "DOMAIN_RELEVANCE_UNVERIFIED"],
        "COCO_2017": ["IMAGE_LEVEL_LICENSE_REVIEW_MISSING", "SUBSET_NOT_SELECTED", "NEGATIVE_ANNOTATION_SCOPE_UNVERIFIED", "DOMAIN_MISMATCH"],
        "OPEN_IMAGES_V7": ["PER_IMAGE_LICENSE_REVIEW_MISSING", "DOWNLOAD_AUTHORIZATION_DENIED", "VERIFIED_NEGATIVE_SUBSET_NOT_SELECTED", "IN_CABIN_DOMAIN_MISMATCH"],
        "ROBOFLOW_SEATBELT_MOBILE": ["ORIGINAL_ASSET_PROVENANCE_MISSING", "AUGMENTATION_LINEAGE_MISSING", "HUMAN_LABEL_QA_MISSING", "SUBJECT_GROUPING_MISSING"],
        "ROBOFLOW_SEATBELT_V4": ["ORIGINAL_ASSET_PROVENANCE_INCOMPLETE", "8371_LABELS_PENDING", "AUGMENTATION_GROUP_LEAKAGE_RISK", "SUBJECT_GROUPING_MISSING"],
        "ROBOFLOW_SEATBELT_THREE_STATE_CANDIDATE": ["ORIGINAL_ASSET_PROVENANCE_MISSING", "EXPORTED_CLASS_MAPPING_AMBIGUOUS", "UNCERTAIN_CLASS_NOT_VERIFIED", "HUMAN_LABEL_QA_MISSING"],
    }
    actions = {
        "DMD_2026": {"permission_needed": "Authorized project record with RIGHTS_ACCEPTED, TRAINING_ALLOWED, LABELS_APPROVED and GROUP_SPLIT_VERIFIED", "action_url": "https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset", "allowed_after_approval": ["local fitting expressly covered by the grant"], "verification_evidence": ["written grant/reference", "updated authoritative permission record", "approved labels", "split receipt"], "expected_contribution": "temporal phone-event truth"},
        "100_DRIVER": {"permission_needed": "Signed research license and written prior consent from PIC Lab", "action_url": "https://100-driver.github.io/img/100_Driver_Licensing.pdf", "allowed_after_approval": ["download", "research action fitting subject to written terms"], "verification_evidence": ["signed form", "approval email/reference", "archive hash", "subject manifest"], "expected_contribution": "cross-driver SAFE/PHONE/OTHER representation"},
        "STATE_FARM_DISTRACTED_DRIVER": {"permission_needed": "User acceptance of Kaggle competition rules in the user's account", "action_url": "https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/rules", "allowed_after_approval": ["download and research use only as allowed by accepted rules"], "verification_evidence": ["rules acceptance receipt", "rules/version snapshot", "driver_imgs_list.csv hash"], "expected_contribution": "subject-disjoint still-image action pretraining"},
        "COCO_2017": {"permission_needed": "Per-image/source-license review for a minimal cell-phone subset", "action_url": "https://cocodataset.org/#download", "allowed_after_approval": ["approved subset acquisition", "generic phone detector pretraining"], "verification_evidence": ["selected image IDs", "license/provenance table", "annotation hash"], "expected_contribution": "generic phone bounding boxes"},
        "OPEN_IMAGES_V7": {"permission_needed": "Per-image license and attribution review for Mobile phone positives and verified negatives", "action_url": "https://storage.googleapis.com/openimages/web/download_v7.html", "allowed_after_approval": ["approved subset acquisition", "phone detector validation/pretraining by declared split"], "verification_evidence": ["image IDs", "license metadata", "box/negative-label hashes", "attribution manifest"], "expected_contribution": "small-phone positives and reviewed hard negatives"},
        "ROBOFLOW_CANDIDATES": {"permission_needed": "Original-image provenance, exact export classes and human QA approval", "action_url": "https://universe.roboflow.com/", "allowed_after_approval": ["quarantined review first", "training only after source-level approval"], "verification_evidence": ["original asset lineage", "license evidence", "duplicate clusters", "human review receipts"], "expected_contribution": "seatbelt and mixed phone candidates"},
    }
    write("dataset_access_action_plan.json", {"source_head": head, "generated_at": datetime.now(UTC).isoformat(), "datasets": [{"dataset_id": item["dataset_id"], "official_source": item["official_source_url"], "status": item["access_status"], "blockers": blockers[item["dataset_id"]]} for item in registry], "user_actions": actions, "automatic_permission_changes": 0})
    write("approved_data_inventory.json", {"existing_rows": inventory["total_records"], "approved_rows": inventory["total_approved_for_training"], "new_v17_original_images": 0, "new_v17_subjects": 0, "new_v17_videos": 0, "new_v17_events": 0, "manifest_breakdown": inventory["existing_ingest_manifests"], "dmd_permissions": permissions, "protected_subjects_accessed": False})
    write("phone_benchmark_protocol.json", {"schema_implementation": "training/dmd/phone_benchmark.py", "status": "BLOCKED_ZERO_APPROVED_RECORDS", "required_truth": {"positive": "Every visible phone has a human-approved pixel xyxy box", "negative": "Human confirms ALL_VISIBLE_PHONES scope and zero boxes", "association": ["DRIVER", "PASSENGER", "UNKNOWN"]}, "required_contexts": ["HANDHELD", "NEAR_EAR", "IN_LAP", "MOUNTED_PHONE", "MOUNTED_CAMERA", "PASSENGER_PHONE", "PHONE_LIKE_OBJECT", "PARTIAL_OCCLUSION", "LIGHTING_AND_CAMERA_DOMAINS"], "current_counts": {"positive_images": 0, "reviewed_negative_images": 0, "boxes": 0}, "ap_metrics_allowed": False})
    write("annotation_qa.json", {"phone_schema": "IMPLEMENTED_AND_TESTED", "candidate_flow_ledger": "IMPLEMENTED_AND_TESTED", "human_approved_phone_records": 0, "inter_annotator_agreement": "NOT_RUN", "adjudication": "NOT_RUN", "seatbelt_pending": 500, "automatic_approvals": 0})
    write("group_split_manifest.json", {"status": "NOT_ESTABLISHED_ZERO_APPROVED_RECORDS", "approved_records": 0, "split_implementation": "training/split_dataset.py", "pre_materialization_checks": ["sha256", "near_cluster_id", "subject", "video", "session", "original_asset_id", "protected_subject"], "train": 0, "validation": 0, "locked_test": 0, "independent": False})
    sliced = read_json("reports/v15_research/sliced_phone_diagnostic.json")
    write("sahi_ablation.json", {"status": "BLOCKED_NO_APPROVED_POSITIVE_NEGATIVE_BENCHMARK", "code": "training/dmd/sliced_phone.py", "code_sha256": sha256_file(ROOT / "training/dmd/sliced_phone.py"), "engineering_diagnostic": sliced, "D0_D4_accuracy_metrics": None, "phone_ap_improved": None, "interpretation": "geometry and overhead verified; accuracy not established"})
    action = metrics["clip_selected_e3"]["pooled"]
    write("action_experiments.json", {"status": "BLOCKED_NO_FITTING_RIGHTS_OR_APPROVED_SPLIT", "experiments_run": 0, "fitted_head_scaler": model["e3_head_scaler"], "inherited_development_only": {"macro_f1": action["macro_f1"], "safe_recall": action["safe_recall"], "phone_recall": action["phone_recall"], "other_recall": action["other_recall"]}, "safe_recall_improved": False})
    config = yaml.safe_load((ROOT / "models/model_config_v4_candidate.yaml").read_text(encoding="utf-8"))["temporal"]
    write("association_fusion_analysis.json", {"status": "INSTRUMENTATION_READY_ACCURACY_BLOCKED", "ledger": "training/dmd/candidate_flow.py", "current_blockers": ["NO_VERIFIED_ROLE_GROUND_TRUTH", "NO_COHORT_WITH_OBJECT_ACTION_EVENT_TRUTH"], "candidate_flow_metrics": None, "needs_review_is_not_true_positive": True})
    write("temporal_event_analysis.json", {"active_candidate_config": config, "short_event_floor_seconds": max(float(config["min_positive_seconds"]), (int(config["min_observations"]) - 1) * 0.0), "finding": f"Events with evidence spans below {float(config['min_positive_seconds']):.2f}s cannot activate under this candidate configuration; accuracy effect requires frozen-event replay.", "config_changed": False, "matcher_changed": False})
    event = metrics["phone_event_controlled_29"]
    write("event_metrics.json", {"scope": "INHERITED_DEVELOPMENT_ONLY_NO_V17_PREDICTION_CHANGE", "tp": event["true_positives"], "fp": event["false_positives"], "fn": event["false_negatives"], "precision": event["precision"], "recall": event["recall"], "f1": event["f1"], "false_alarms_per_minute": event["false_alarms_per_minute"], "delta_tp": 0, "delta_fp": 0, "delta_fn": 0, "independent": False})
    seatbelt = metrics["seatbelt"]
    write("seatbelt_review_status.json", {"status": "BLOCKED_500_PENDING_HUMAN_REVIEWS", "pending": 500, "automatic_approvals": 0, "best_inherited_development_macro_f1": seatbelt["best_development_macro_f1"], "uncertain_recall": seatbelt["variants"][2]["uncertain_recall"], "new_metrics": None})
    write("experiment_registry.json", {"source_head": head, "experiments": [{"id": "V17-E1-SPLIT-HARDENING", "hypothesis": "component grouping prevents subject/video/original leakage", "decision": "INCONCLUSIVE", "reason": "engineering tests pass; no approved real split exists"}, {"id": "V17-E2-PHONE-BENCHMARK-CONTRACT", "hypothesis": "explicit reviewed negatives prevent false background truth", "decision": "INCONCLUSIVE", "reason": "contract tests pass; zero approved benchmark records"}, {"id": "V17-E3-CANDIDATE-FLOW-LEDGER", "hypothesis": "stage reason codes expose recall loss", "decision": "INCONCLUSIVE", "reason": "instrumentation tests pass; no role/object/event GT cohort"}], "training_performed": False, "production_changed": False})
    write("model_manifest.json", {**model, "v17_fitted_models": [], "active_model_changed": False, "production_approved": False})
    write("environment.json", {"generated_at": datetime.now(UTC).isoformat(), "source_head": head, "inventory_hash": stable_json_hash(inventory), "registry_hash": stable_json_hash(registry)})


if __name__ == "__main__":
    run()
