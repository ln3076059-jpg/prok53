"""V14 autonomous-path audit without fitting or oracle inputs."""

from __future__ import annotations

import json
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

from training.common import sha256_file
from training.dmd.autonomous_phone import YoloPhoneDetector
from training.dmd.research_permissions import read_permissions

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports/v14_research"


def load(path: str) -> dict:
    return json.loads((ROOT / path).read_text(encoding="utf-8"))


def write(name: str, value: object) -> None:
    (OUT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def run() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    rights = read_permissions()
    historical = load("reports/v9_research/phone_proposal_mining.json")
    phone_path = ROOT / historical["models"]["specialist"]["path"]
    detector = YoloPhoneDetector(phone_path)
    if detector.model_sha256 != historical["models"]["specialist"]["sha256"]:
        raise ValueError("specialist phone detector hash changed")
    auxiliary = ROOT / historical["models"]["coco_cell_phone"]["path"]
    if sha256_file(auxiliary) != historical["models"]["coco_cell_phone"]["sha256"]:
        raise ValueError("auxiliary detector hash changed")
    auto = {
        "mode": "AUTONOMOUS_DETECTION_EVALUATION",
        "new_v14_runtime_execution": "NOT_RUN_RESOURCE_CONSTRAINED_RECONCILED_EXISTING_RUN",
        "source": "reports/v9_research/phone_proposal_mining.json",
        "source_sha256": sha256_file(ROOT / "reports/v9_research/phone_proposal_mining.json"),
        "input_phone_boxes": "DETECTOR_RUNTIME_ONLY",
        "reviewed_boxes_used_as_input": False,
        "summary": historical["summary"],
        "confidence_floor": historical["confidence_floor"],
        "models": historical["models"],
        "phone_ap": None,
        "precision": None,
        "proposal_recall": None,
        "limitations": historical["limitations"],
    }
    write("autonomous_proposal_audit.json", auto)
    v13_data = load("reports/v13_research/dataset_manifest.json")
    v13_data.update(
        v14_audit_head=head,
        permission_sha256=sha256_file(ROOT / "dataset/external_dmd/README.md"),
        live_permissions={key: rights.get(key) for key in (
            "RIGHTS_STATUS", "RIGHTS_ACCEPTED", "CANONICAL_ELIGIBLE",
            "TEMPORAL_DEVELOPMENT_ELIGIBLE", "UNTOUCHED_HOLDOUT_ELIGIBLE"
        )},
        protected_media_or_labels_opened=False,
    )
    write("data_manifest.json", v13_data)
    write("model_manifest.json", {
        "phone_detector": {
            "path": phone_path.relative_to(ROOT).as_posix(),
            "sha256": detector.model_sha256,
        },
        "auxiliary_coco": {
            "path": auxiliary.relative_to(ROOT).as_posix(),
            "sha256": sha256_file(auxiliary),
        },
        "e3_head_scaler": None,
        "active_model_changed": False,
    })
    metrics = load("reports/v13_research/metrics_comparison.json")
    metrics.update(
        v14_new_event_metrics=None,
        autonomous_phone_ap=None,
        independent_ai85_established=False,
    )
    write("metrics_comparison.json", metrics)
    readiness = {
        "autonomous_adapter": "READY_ENGINEERING",
        "specialist_detector_artifact": "HASH_VERIFIED_RESEARCH_ONLY",
        "autonomous_accuracy_evaluation": "BLOCKED_NO_BBOX_NEGATIVE_GROUND_TRUTH",
        "online_e3_artifact_contract_v2": "READY_SYNTHETICALLY_TESTED",
        "fitted_e3_head_scaler": "BLOCKED_MISSING",
        "verified_driver_association": "BLOCKED_MISSING_GROUND_TRUTH",
        "real_event_replay": "BLOCKED_MISSING_FITTED_HEAD_AND_VERIFIED_ASSOCIATION",
        "seatbelt": "BLOCKED_500_PENDING_LABELS",
        "independent_evaluation": "BLOCKED_NO_NEW_LOCKED_HOLDOUT",
        "production": "REJECT",
    }
    write("candidate_readiness.json", readiness)
    write("event_match_records.json", {
        "source": "reports/v13_research/event_match_records.json",
        "source_sha256": sha256_file(ROOT / "reports/v13_research/event_match_records.json"),
        "v14_real_candidate_replay": readiness["real_event_replay"],
        "ground_truth_changed": False,
        "matcher_changed": False,
    })
    decisions = {
        "E0": "KEEP", "E1": "KEEP", "E2": "KEEP_ENGINEERING_INCONCLUSIVE_ACCURACY",
        "E3": "KEEP_ENGINEERING", "E4": "BLOCKED", "E5": "BLOCKED",
        "E6": "BLOCKED", "E7": "BLOCKED", "E8": "BLOCKED",
    }
    write("experiment_registry.json", {
        "source_head": head,
        "plan_sha256": sha256_file(OUT / "experiment_plan.json"),
        "decisions": decisions,
        "training_performed": False,
        "protected_holdouts_accessed": False,
        "production_approved": False,
    })
    write("environment.json", {
        "generated_at": datetime.now(UTC).isoformat(), "head": head,
        "python": platform.python_version(), "gpu": "NOT_AVAILABLE_NVIDIA_SMI",
        "runtime_note": (
            "New YOLO execution skipped because prior audit found constrained RAM; "
            "immutable existing runtime proposals reconciled"
        ),
    })


if __name__ == "__main__":
    run()
