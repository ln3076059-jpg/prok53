from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.build_adt_review_queue import build as build_adt_queue
from training.common import sha256_file
from training.merge_phone_proposals import merge as merge_phone_proposals


def _load(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    return payload if isinstance(payload, list) else payload["selected"]


def _one_per_group(items: list[dict]) -> list[dict]:
    selected = {}
    for item in items:
        group = str(item["source_group_id"])
        current = selected.get(group)
        if current is None or item["sample_id"] < current["sample_id"]:
            selected[group] = item
    return [selected[group] for group in sorted(selected)]


def _write_queue(path: Path, items: list[dict]) -> dict:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(items, indent=2, sort_keys=True), encoding="utf-8")
    return {"path": str(path), "samples": len(items), "sha256": sha256_file(path)}


def prepare(output: Path) -> dict:
    dms = _load(Path("datasets/manifests/review_queue_dms.json"))
    phone_positive = _one_per_group([item for item in dms if item.get("annotations")])
    phone_negative = _one_per_group([item for item in dms if not item.get("annotations")])
    seatbelt_source = _load(
        Path("datasets/manifests/review_queue_roboflow_seatbelttraining_v4.json")
    )
    seatbelt_records = {
        item["sample_id"]: item
        for line in Path("datasets/derived/seatbelt_v2_balanced/manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
        for item in [json.loads(line)]
    }
    seatbelt_manifest = set(seatbelt_records)
    uncertain_payload = json.loads(
        Path("datasets/manifests/seatbelt_uncertain_review_v2.json").read_text(encoding="utf-8")
    )
    uncertain_ui = []
    for item in uncertain_payload["selected"]:
        state = seatbelt_records[item["sample_id"]]["seatbelt_states"][0]
        class_id = 1 if state == "seatbelt_fastened" else 2
        uncertain_ui.append(
            {
                **item,
                "candidate_id": item["candidate_id"],
                "source_sample_id": item["sample_id"],
                "sample_id": item["candidate_id"],
                "source_id": "roboflow_seatbelttraining_v4",
                "source_asset_id": item["sample_id"],
                "license": "CC-BY-4.0",
                "occupant_role": "PENDING",
                "vehicle_context_id": None,
                "annotations": [
                    {
                        "class_id": class_id,
                        "occupant_role": "PENDING",
                        "requires_review": True,
                        "yolo": item["yolo"],
                    }
                ],
                "notes": (
                    "Choose Uncertain when belt evidence is invisible, occluded, dark, blurred, "
                    "or contradictory. Approve boxes only for positive visible evidence."
                ),
            }
        )
    seatbelt = [item for item in seatbelt_source if item["sample_id"] in seatbelt_manifest]
    if len(seatbelt) != len(seatbelt_manifest):
        raise ValueError("seatbelt review queue does not cover the balanced V2 manifest")

    queues = {
        "phone_positive_group_representatives": _write_queue(
            Path("datasets/manifests/v2_phone_positive_review.json"), phone_positive
        ),
        "phone_negative_group_representatives": _write_queue(
            Path("datasets/manifests/v2_phone_negative_review.json"), phone_negative
        ),
        "seatbelt_roi_state_all_selected": _write_queue(
            Path("datasets/manifests/v2_seatbelt_roi_state_review.json"), seatbelt
        ),
        "seatbelt_uncertain_ui": _write_queue(
            Path("datasets/manifests/v2_seatbelt_uncertain_ui_review.json"), uncertain_ui
        ),
    }
    mendeley_queue = _load(Path("datasets/manifests/review_queue_mendeley_driver_risk_v1.json"))
    mendeley_proposals = json.loads(
        Path("datasets/manifests/phone_proposals_mendeley_driver_risk_v1.json").read_text(
            encoding="utf-8"
        )
    )
    mendeley_external = merge_phone_proposals(mendeley_queue, mendeley_proposals)
    for item in mendeley_external:
        item["intended_use"] = "PHONE_SOURCE_DISJOINT_EXTERNAL_TEST_CANDIDATE"
        item["split_policy"] = "EXTERNAL_TEST_ONLY_UNTIL_MODEL_LOCK"
    queues["phone_external_mendeley_review"] = _write_queue(
        Path("datasets/manifests/v2_phone_external_mendeley_review.json"), mendeley_external
    )
    adt = build_adt_queue(
        Path("datasets/raw/roboflow_adt_seatbelt_v1/1"),
        Path("datasets/manifests/review_queue_adt_seatbelt_v1.json"),
        Path("datasets/manifests/ingest_adt_seatbelt_v1.jsonl"),
        Path("reports/adt_seatbelt_v1/review_source_audit.json"),
    )
    queues["adt_external_group_representatives"] = {
        "path": "datasets/manifests/review_queue_adt_seatbelt_v1.json",
        "samples": adt["review_representatives"],
        "sha256": adt["queue_sha256"],
    }
    ai_readiness_path = Path("reports/v2_pretrain_pending_approval_readiness.json")
    ai_readiness = (
        json.loads(ai_readiness_path.read_text(encoding="utf-8"))
        if ai_readiness_path.is_file()
        else {}
    )
    ai_state = ai_readiness.get("readiness", {})
    plan = {
        "schema_version": 1,
        "status": "HUMAN_ACTION_REQUIRED_BEFORE_GOVERNED_TRAINING",
        "training_lanes": {
            "pretrain_pending_exploratory": {
                "training_ready": bool(
                    ai_readiness.get("status") == "PASS"
                    and ai_state.get("exploratory_training_ready")
                ),
                "governance_level": "MODEL_ASSISTED_PENDING_APPROVAL",
                "human_verified": False,
                "governed_training_ready": False,
                "production_activation_ready": False,
                "readiness_report": "reports/v2_pretrain_pending_approval_readiness.json",
            },
            "human_verified_governed": {
                "training_ready": False,
                "human_action_required": True,
                "readiness_report": "reports/v2_training_readiness.json",
            },
        },
        "queues": queues,
        "review_order": [
            "v2_seatbelt_uncertain_ui_review",
            "v2_phone_negative_review",
            "v2_phone_positive_review",
            "v2_phone_external_mendeley_review",
            "v2_seatbelt_roi_state_review",
            "review_queue_adt_seatbelt_v1",
        ],
        "commands": {
            "run_proposal_review": "py -m training.prepare_v2_pretrain_review",
            "build_pretrain_pending_datasets": "py -m training.build_pretrain_pending_v2",
            "audit_pretrain_pending_datasets": "py -m training.audit_pretrain_pending_v2",
            "build_pretrain_pending_bundle": "py -m training.build_v2_pretrain_bundle",
            "train_pretrain_pending_bundle": ".\\START_V2_PRETRAIN_REVIEW_TRAINING.ps1",
            "review_template": (
                "py tools/annotation_reviewer/app.py --manifest <QUEUE.json> "
                "--decisions <DECISIONS.jsonl> --pending-only"
            ),
            "materialize_phone_negatives": (
                "py -m training.apply_review_decisions datasets/manifests/ingest_dms.jsonl "
                "--decisions datasets/manifests/v2_phone_negative_decisions.jsonl --output "
                "datasets/reviewed/v2_phone --manifest "
                "datasets/manifests/reviewed_v2_phone_negative.jsonl"
            ),
            "materialize_phone_positives": (
                "py -m training.apply_review_decisions datasets/manifests/ingest_dms.jsonl "
                "--decisions datasets/manifests/v2_phone_positive_decisions.jsonl --output "
                "datasets/reviewed/v2_phone --manifest "
                "datasets/manifests/reviewed_v2_phone_positive.jsonl"
            ),
            "materialize_seatbelt_roi_state": (
                "py -m training.apply_review_decisions "
                "datasets/manifests/ingest_roboflow_seatbelttraining_v4.jsonl --decisions "
                "datasets/manifests/v2_seatbelt_roi_state_decisions.jsonl --output "
                "datasets/reviewed/v2_seatbelt --manifest "
                "datasets/manifests/reviewed_v2_seatbelt_roi_state.jsonl"
            ),
            "materialize_phone_external_mendeley": (
                "py -m training.apply_review_decisions "
                "datasets/manifests/ingest_mendeley_driver_risk_v1.jsonl --decisions "
                "datasets/manifests/v2_phone_external_mendeley_decisions.jsonl --output "
                "datasets/reviewed/mendeley_phone_external --manifest "
                "datasets/manifests/reviewed_v2_phone_external_mendeley.jsonl"
            ),
            "materialize_adt_external": (
                "py -m training.apply_review_decisions "
                "datasets/manifests/ingest_adt_seatbelt_v1.jsonl --decisions "
                "datasets/manifests/review_decisions_adt_seatbelt_v1.jsonl --output "
                "datasets/reviewed/adt_external --manifest "
                "datasets/manifests/reviewed_adt_external.jsonl"
            ),
            "refresh_readiness": "py -m training.audit_v2_readiness",
            "convert_uncertain_decisions": (
                "py -m training.convert_uncertain_review_decisions "
                "datasets/manifests/v2_seatbelt_uncertain_ui_decisions.jsonl "
                "--reviewer <NAME>"
            ),
            "materialize_uncertain": (
                "py -m training.apply_uncertain_decisions "
                "datasets/manifests/seatbelt_uncertain_decisions_v2.jsonl"
            ),
            "review_uncertain_ui": (
                "py tools/annotation_reviewer/app.py --manifest "
                "datasets/manifests/v2_seatbelt_uncertain_ui_review.json --decisions "
                "datasets/manifests/v2_seatbelt_uncertain_ui_decisions.jsonl --pending-only"
            ),
            "review_phone_negatives": (
                "py tools/annotation_reviewer/app.py --manifest "
                "datasets/manifests/v2_phone_negative_review.json --decisions "
                "datasets/manifests/v2_phone_negative_decisions.jsonl --pending-only"
            ),
            "review_phone_positives": (
                "py tools/annotation_reviewer/app.py --manifest "
                "datasets/manifests/v2_phone_positive_review.json --decisions "
                "datasets/manifests/v2_phone_positive_decisions.jsonl --pending-only"
            ),
            "review_seatbelt_roi_state": (
                "py tools/annotation_reviewer/app.py --manifest "
                "datasets/manifests/v2_seatbelt_roi_state_review.json --decisions "
                "datasets/manifests/v2_seatbelt_roi_state_decisions.jsonl --pending-only"
            ),
            "review_phone_external_mendeley": (
                "py tools/annotation_reviewer/app.py --manifest "
                "datasets/manifests/v2_phone_external_mendeley_review.json --decisions "
                "datasets/manifests/v2_phone_external_mendeley_decisions.jsonl --pending-only"
            ),
            "review_adt_external": (
                "py tools/annotation_reviewer/app.py --manifest "
                "datasets/manifests/review_queue_adt_seatbelt_v1.json --decisions "
                "datasets/manifests/review_decisions_adt_seatbelt_v1.jsonl --pending-only"
            ),
        },
        "policy": (
            "Only explicitly reviewed rows may enter a governed dataset. Group representatives "
            "reduce augmentation duplication but never approve unseen variants."
        ),
        "external_candidates": {
            "mendeley_driver_risk_behavior_v1": {
                "task": "PHONE_DETECTION",
                "license": "CC-BY-4.0",
                "source_images": len(mendeley_queue),
                "selected_phone_behavior_frames": len(mendeley_external),
                "inferred_sequence_groups": len(
                    {item["source_group_id"] for item in mendeley_external}
                ),
                "automatic_training_admission": False,
                "split_policy": "EXTERNAL_TEST_ONLY_UNTIL_MODEL_LOCK",
            },
            "roboflow_adt_seatbelt_v1": {
                "task": "SEATBELT_AND_PHONE_EXTERNAL_EVALUATION",
                "license": "CC-BY-4.0",
                "source_images": adt["source_images"],
                "inferred_groups": adt["inferred_groups"],
                "automatic_training_admission": False,
                "split_policy": "EXTERNAL_TEST_ONLY_UNTIL_MODEL_LOCK",
            },
        },
        "external_test_policy": (
            "Reviewed Mendeley phone frames and ADT representatives are reserved as "
            "source-disjoint external test candidates and must not tune hyperparameters or "
            "thresholds."
        ),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(plan, indent=2, sort_keys=True), encoding="utf-8")
    return plan


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare all V2 human-review work for data gaps")
    parser.add_argument("--output", type=Path, default=Path("reports/v2_data_gap_action_plan.json"))
    args = parser.parse_args()
    print(json.dumps(prepare(args.output), indent=2))


if __name__ == "__main__":
    main()
