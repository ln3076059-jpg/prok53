"""V16 data-centric registry, local inventory, and fail-closed readiness audit."""

# ruff: noqa: E501

from __future__ import annotations

import json
import platform
import subprocess
from collections import Counter
from datetime import UTC, datetime
from pathlib import Path

import yaml

from training.common import sha256_file, stable_json_hash
from training.dmd.research_permissions import read_permissions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v16_data_ai85"


def read_json(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def registry() -> list[dict]:
    common = {
        "duplicate_status": "NOT_EVALUATED_NO_V16_DOWNLOAD",
        "training_eligibility": False,
        "validation_eligibility": False,
        "independent_evaluation_eligibility": False,
    }
    rows = [
        {
            "dataset_id": "DMD_2026",
            "official_source_url": "https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset",
            "version": "2026 revision",
            "source_owner": "Vicomtech",
            "license": "CC BY-NC-ND 4.0",
            "access_status": "BLOCKED",
            "allowed_operations": ["local_temporal_diagnostics_on_existing_development_media"],
            "download_size": "NOT VERIFIED",
            "original_samples": "NOT RECOUNTED_FROM_REMOTE",
            "augmented_samples": 0,
            "subjects": "14 unrestricted IDs listed by owner; project rights remain denied",
            "cameras": 3,
            "annotation_format": "OpenLABEL/VCD >=5.0",
            "annotation_quality": "official temporal labels; no complete phone bbox benchmark",
            "tasks_supported": ["driver action", "temporal event diagnostics"],
            "target_labels": ["SAFE_DRIVING", "TEXTING", "HANDHELD_CALL"],
            "domain": "real-car RGB BODY/FACE/HANDS",
            "known_limitations": ["RIGHTS_ACCEPTED=false", "NoDerivatives redistribution restriction"],
            "provenance_hash": None,
            **common,
        },
        {
            "dataset_id": "100_DRIVER",
            "official_source_url": "https://100-driver.github.io/",
            "version": "official current page",
            "source_owner": "PIC Lab, Chinese Academy of Sciences",
            "license": "research-only custom license; prior consent required",
            "access_status": "ACCESS_REQUEST_REQUIRED",
            "allowed_operations": [],
            "download_size": "NOT VERIFIED",
            "original_samples": "more than 470,000 reported",
            "augmented_samples": "NOT VERIFIED",
            "subjects": 100,
            "cameras": 4,
            "annotation_format": "image classification/splits",
            "annotation_quality": "official classification benchmark; no event timestamps or phone boxes",
            "tasks_supported": ["cross-driver action classification", "cross-view robustness"],
            "target_labels": ["action labels require mapping review"],
            "domain": "5 vehicles, multiple views/lighting/modalities",
            "known_limitations": ["signed form and prior consent required", "static images are not events"],
            "provenance_hash": None,
            **common,
        },
        {
            "dataset_id": "STATE_FARM_DISTRACTED_DRIVER",
            "official_source_url": "https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data",
            "version": "competition",
            "source_owner": "State Farm/Kaggle competition",
            "license": "SUBJECT_TO_COMPETITION_RULES",
            "access_status": "ACCESS_REQUEST_REQUIRED",
            "allowed_operations": [],
            "download_size": "NOT VERIFIED",
            "original_samples": "NOT VERIFIED IN CURRENT ACCESS",
            "augmented_samples": 0,
            "subjects": "available through driver_imgs_list.csv after authorized access",
            "cameras": "NOT VERIFIED",
            "annotation_format": "image class plus driver mapping CSV",
            "annotation_quality": "action class only; no physical-phone boxes or event intervals",
            "tasks_supported": ["action pretraining", "hard negatives"],
            "target_labels": ["c0 SAFE", "c1-c4 PHONE", "c5-c9 OTHER"],
            "domain": "in-cabin still images",
            "known_limitations": ["competition rules require acceptance", "subject split mandatory"],
            "provenance_hash": None,
            **common,
        },
        {
            "dataset_id": "DRIVE_AND_ACT",
            "official_source_url": "https://www.driveandact.com/",
            "version": "site updated 2021-05-27",
            "source_owner": "Fraunhofer IOSB/KIT",
            "license": "research use only",
            "access_status": "DISCOVERED",
            "allowed_operations": ["metadata_research"],
            "download_size": "NOT VERIFIED",
            "original_samples": "12 hours, 29 long sequences",
            "augmented_samples": 0,
            "subjects": "NOT VERIFIED ON LANDING PAGE",
            "cameras": 5,
            "annotation_format": "83 hierarchical temporal activity labels and pose",
            "annotation_quality": "official activity labels; phone equivalence not established",
            "tasks_supported": ["multi-view temporal representation", "object interaction"],
            "target_labels": ["none mapped automatically"],
            "domain": "automated vehicle NIR/depth/RGB",
            "known_limitations": ["phone label not found on landing page", "domain differs"],
            "provenance_hash": None,
            **common,
        },
        {
            "dataset_id": "COCO_2017",
            "official_source_url": "https://cocodataset.org/",
            "version": "2017 detection",
            "source_owner": "COCO Consortium; images retain source copyrights",
            "license": "annotations and per-image terms require separate verification",
            "access_status": "LICENSE_UNVERIFIED",
            "allowed_operations": ["metadata_research"],
            "download_size": "full dataset intentionally not requested",
            "original_samples": "NOT RECOUNTED",
            "augmented_samples": 0,
            "subjects": "not applicable/unknown",
            "cameras": "many web image sources",
            "annotation_format": "COCO JSON boxes",
            "annotation_quality": "object boxes; negatives need annotation-scope care",
            "tasks_supported": ["generic mobile-phone detection pretraining"],
            "target_labels": ["cell phone category; exact numeric ID to verify in export"],
            "domain": "generic images",
            "known_limitations": ["domain mismatch", "image copyrights vary"],
            "provenance_hash": None,
            **common,
        },
        {
            "dataset_id": "OPEN_IMAGES_V7",
            "official_source_url": "https://storage.googleapis.com/openimages/web/download_v7.html",
            "version": "V7, October 2022",
            "source_owner": "Google/Open Images; underlying images retain individual licenses",
            "license": "per-image license in image metadata; annotation terms require attribution review",
            "access_status": "DISCOVERED",
            "allowed_operations": ["metadata_research"],
            "download_size": "subset not selected",
            "original_samples": "1.9M dense-annotation subset; 9,178,275 total reported",
            "augmented_samples": 0,
            "subjects": "not provided",
            "cameras": "web image sources",
            "annotation_format": "CSV normalized boxes/image labels",
            "annotation_quality": "xclick or human-verified activemil; absence must use verified negatives",
            "tasks_supported": ["generic phone detection", "hard negatives"],
            "target_labels": ["/m/050k8 Mobile phone"],
            "domain": "generic web imagery",
            "known_limitations": ["incomplete boxes outside dense subset", "per-image licenses"],
            "provenance_hash": "1839e0e7e84130ae281f7f67413768601b031581c0c42e7fc17527b8e2a99aa9",
            **common,
        },
        {
            "dataset_id": "ROBOFLOW_SEATBELT_MOBILE",
            "official_source_url": "https://universe.roboflow.com/naina-iucjd/seatbelt_and_mobile_detection-1qojv",
            "version": 1,
            "source_owner": "naina / Roboflow Universe",
            "license": "CC BY 4.0 shown on project page; original-image provenance unverified",
            "access_status": "QUARANTINED_FOR_REVIEW",
            "allowed_operations": ["metadata_research"],
            "download_size": "NOT VERIFIED",
            "original_samples": "6,547 shown on overview; model dataset says 15,877 generated images",
            "augmented_samples": "NOT VERIFIED",
            "subjects": "NOT VERIFIED",
            "cameras": "mixed roadside/cabin, NOT VERIFIED",
            "annotation_format": "object detection",
            "annotation_quality": "classes visible; source/augmentation lineage not established",
            "tasks_supported": ["phone/seatbelt candidate review"],
            "target_labels": ["mobile", "seatbelt", "no-seatbelt"],
            "domain": "mixed vehicle images",
            "known_limitations": ["no uncertainty class", "published model metrics are external"],
            "provenance_hash": None,
            **common,
        },
        {
            "dataset_id": "ROBOFLOW_SEATBELT_V4",
            "official_source_url": "https://universe.roboflow.com/seatbelttraining-7yh0f/seatbelt-detection-lb1ec",
            "version": 4,
            "source_owner": "seatbelttraining / Roboflow Universe",
            "license": "CC BY 4.0 shown on project page; original-image provenance under review",
            "access_status": "ANNOTATION_PENDING",
            "allowed_operations": ["local_quarantine_review"],
            "download_size": "already present locally; no V16 download",
            "original_samples": "3,489 overview images",
            "augmented_samples": "8,371 generated export images",
            "subjects": "NOT VERIFIED",
            "cameras": "mixed",
            "annotation_format": "YOLO boxes",
            "annotation_quality": "published groups leak; all 8,371 local records pending",
            "tasks_supported": ["seatbelt candidate review"],
            "target_labels": ["seatbelt", "no-seatbelt"],
            "domain": "mixed/grayscale augmented cabin imagery",
            "known_limitations": ["no UNCERTAIN", "augmentation grouping leakage", "provenance review"],
            "provenance_hash": None,
            **common,
        },
        {
            "dataset_id": "ROBOFLOW_SEATBELT_THREE_STATE_CANDIDATE",
            "official_source_url": "https://universe.roboflow.com/eco3cc/seat-belt-detection-uhqwa-0tqd0",
            "version": "project version 8; model points to dataset version 1",
            "source_owner": "Eco3cc / Roboflow Universe",
            "license": "CC BY 4.0 shown on project page; original-image provenance unverified",
            "access_status": "QUARANTINED_FOR_REVIEW",
            "allowed_operations": ["metadata_research"],
            "download_size": "NOT VERIFIED",
            "original_samples": 660,
            "augmented_samples": "NOT VERIFIED",
            "subjects": "NOT VERIFIED",
            "cameras": "NOT VERIFIED",
            "annotation_format": "object detection",
            "annotation_quality": "description mentions Not Clear but exported class list is five ambiguous labels",
            "tasks_supported": ["three-state schema investigation"],
            "target_labels": ["seatbelt", "1", "2", "person-noseatbelt", "person-seatbelt"],
            "domain": "vehicle images",
            "known_limitations": ["Not Clear is not a verified export class", "annotation ongoing"],
            "provenance_hash": None,
            **common,
        },
    ]
    return rows


def inventory_manifests() -> list[dict]:
    rows = []
    for path in sorted((ROOT / "dataset/manifests").glob("ingest_*.jsonl")):
        count, annotations = 0, 0
        hashes, groups = set(), set()
        statuses: Counter[str] = Counter()
        with path.open(encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                item = json.loads(line)
                count += 1
                annotations += int(item.get("annotation_count", 0))
                hashes.add(item.get("sha256"))
                groups.add(item.get("source_group_id"))
                statuses[str(item.get("review_status", "MISSING"))] += 1
        rows.append(
            {
                "manifest": path.relative_to(ROOT).as_posix(),
                "sha256": sha256_file(path),
                "records": count,
                "unique_sha256": len(hashes),
                "source_groups": len(groups),
                "source_annotations": annotations,
                "review_status": dict(statuses),
                "approved_for_training": statuses.get("APPROVED", 0),
            }
        )
    return rows


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    sources_path = ROOT / "dataset/sources.yaml"
    sources = yaml.safe_load(sources_path.read_text(encoding="utf-8"))
    entries = registry()
    write(
        "dataset_registry.json",
        {
            "schema_version": 1,
            "generated_at": datetime.now(UTC).isoformat(),
            "source_head": head,
            "registry": entries,
            "selected_for_v16_download": [],
            "v16_downloads_performed": 0,
        },
    )
    write(
        "license_audit.json",
        {
            "verified_on": "2026-10-18",
            "sources": [
                {
                    "dataset_id": item["dataset_id"],
                    "license": item["license"],
                    "access_status": item["access_status"],
                    "allowed_operations": item["allowed_operations"],
                    "source_url": item["official_source_url"],
                }
                for item in entries
            ],
            "dmd_live_permissions": read_permissions(),
            "local_registry_source": "dataset/sources.yaml",
            "local_registry_sha256": sha256_file(sources_path),
            "legacy_registry_entries": len(sources["sources"]),
        },
    )
    local = inventory_manifests()
    write(
        "data_manifest.json",
        {
            "existing_ingest_manifests": local,
            "total_records": sum(item["records"] for item in local),
            "total_approved_for_training": sum(item["approved_for_training"] for item in local),
            "new_v16_original_images": 0,
            "new_v16_subjects": 0,
            "new_v16_videos": 0,
            "new_v16_events": 0,
            "raw_media_committed": False,
            "protected_subjects_accessed": False,
        },
    )
    mappings = {
        "schema_version": 1,
        "mapping_version": "ROADWATCH_UNIFIED_V16_1",
        "unified_labels": {
            "phone_visibility": ["PHONE_BOX", "PHONE_VISIBLE", "PHONE_OCCLUDED", "PHONE_ABSENT", "UNKNOWN_PHONE_VISIBILITY"],
            "phone_action": ["SAFE_DRIVING", "TEXTING", "HANDHELD_CALL", "OTHER_PHONE_USE", "OTHER_ACTIVITY", "UNKNOWN_ACTION"],
            "phone_context": ["HANDHELD", "MOUNTED", "STATIC_CAMERA", "PASSENGER_PHONE", "UNKNOWN_CONTEXT"],
            "occupant_role": ["DRIVER", "FRONT_PASSENGER", "REAR_PASSENGER", "UNKNOWN"],
            "seatbelt": ["FASTENED", "UNFASTENED", "UNCERTAIN_OR_OCCLUDED"],
        },
        "state_farm_proposed": {"c0": "SAFE_DRIVING", "c1": "TEXTING", "c2": "HANDHELD_CALL", "c3": "TEXTING", "c4": "HANDHELD_CALL", "c5-c9": "OTHER_ACTIVITY"},
        "rules": ["retain original label", "static images have no event interval", "UNKNOWN is never SAFE", "visible phone is not phone use", "invisible belt is not UNFASTENED"],
    }
    write("label_mapping.json", mappings)
    write(
        "annotation_qa.json",
        {
            "written_schema": "training/data_governance.py",
            "bbox_validation": "IMPLEMENTED_AND_SYNTHETICALLY_TESTED",
            "reviewer_requirement": "IMPLEMENTED_FOR_APPROVED_LABELS",
            "inter_annotator_agreement": "NOT_RUN_NO_NEW_DOUBLE_ANNOTATED_SUBSET",
            "disagreement_adjudication": "REQUIRED_NOT_EXECUTED",
            "seatbelt_pending_candidates": 500,
            "automatic_approvals": 0,
            "phone_detection_benchmark": "BLOCKED_NO_APPROVED_POSITIVE_AND_REVIEWED_NEGATIVE_SET",
        },
    )
    governed = ROOT / "dataset/manifests/v2_governed.jsonl"
    write(
        "group_split_manifest.json",
        {
            "governed_manifest": governed.relative_to(ROOT).as_posix(),
            "governed_manifest_sha256": sha256_file(governed),
            "approved_records": 0,
            "train_groups": 0,
            "validation_groups": 0,
            "locked_test_groups": 0,
            "group_disjoint": "NOT_ESTABLISHED_NO_APPROVED_RECORDS",
            "independent_evaluation": False,
            "protected_subjects": ["gZ-37", "gE-28"],
        },
    )
    metrics = read_json("reports/v15_research/metrics_comparison.json")
    metrics.update(
        v16_new_phone_ap=None,
        v16_new_event_metrics=None,
        v16_new_action_metrics=None,
        v16_new_seatbelt_metrics=None,
        v16_downloads=0,
        independent_ai85_established=False,
        production_approved=False,
    )
    write("metrics_comparison.json", metrics)
    models = read_json("reports/v15_research/model_manifest.json")
    models.update(v16_fitted_models=[], active_model_changed=False, production_approved=False)
    write("model_manifest.json", models)
    write(
        "candidate_readiness.json",
        {
            "dataset_registry": "READY",
            "download_gate": "READY_FAIL_CLOSED",
            "unified_annotation_schema": "READY_ENGINEERING",
            "approved_training_data": "BLOCKED_ZERO_APPROVED_RECORDS",
            "phone_detection_ablation": "BLOCKED_NO_APPROVED_BBOX_POSITIVE_NEGATIVE_VALIDATION",
            "action_fitting": "BLOCKED_RIGHTS_AND_GROUP_SPLIT",
            "association_evaluation": "BLOCKED_NO_VERIFIED_ROLE_GT",
            "event_replay": "BLOCKED_NO_NEW_VALIDATED_CANDIDATE",
            "seatbelt_fitting": "BLOCKED_500_PENDING_LABELS",
            "independent_evaluation": "BLOCKED_NO_LOCKED_TEST",
            "production": "REJECT",
        },
    )
    write(
        "experiment_registry.json",
        {
            "source_head": head,
            "V16_D0_D4": "BLOCKED_BEFORE_INFERENCE_NO_APPROVED_VALIDATION_PROTOCOL",
            "V16_A0_A6": "BLOCKED_NO_FITTING_RIGHTS",
            "V16_TEMPORAL": "NOT_RUN_NO_NEW_CANDIDATE",
            "V16_SEATBELT": "BLOCKED_PENDING_HUMAN_REVIEW",
            "downloads_performed": 0,
            "training_performed": False,
            "protected_holdouts_accessed": False,
            "production_approved": False,
        },
    )
    write(
        "environment.json",
        {
            "generated_at": datetime.now(UTC).isoformat(),
            "source_head": head,
            "python": platform.python_version(),
            "platform": platform.platform(),
            "registry_sha256": stable_json_hash(entries),
        },
    )


if __name__ == "__main__":
    run()
