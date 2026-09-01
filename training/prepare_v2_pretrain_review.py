from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.common import sha256_file
from training.pretrain_review_v2 import MIN_UNCERTAIN_COUNTS, REVIEWER_MODEL, run
from training.uncertain_visual_decisions import load_selected_uncertain_rows

SPECS = {
    "phone_positive": {
        "queue": Path("datasets/manifests/v2_phone_positive_review.json"),
        "output": Path("datasets/manifests/pretrain_pending_approval/v2_phone_positive.json"),
        "task": "phone",
        "phone_model": True,
        "core": True,
    },
    "phone_negative": {
        "queue": Path("datasets/manifests/v2_phone_negative_review.json"),
        "output": Path("datasets/manifests/pretrain_pending_approval/v2_phone_negative.json"),
        "task": "phone",
        "phone_model": True,
        "core": True,
    },
    "phone_external_mendeley": {
        "queue": Path("datasets/manifests/v2_phone_external_mendeley_review.json"),
        "output": Path(
            "datasets/manifests/pretrain_pending_approval/v2_phone_external_mendeley.json"
        ),
        "task": "phone",
        "phone_model": True,
        "core": False,
    },
    "seatbelt_roi_state": {
        "queue": Path("datasets/manifests/v2_seatbelt_roi_state_review.json"),
        "output": Path("datasets/manifests/pretrain_pending_approval/v2_seatbelt_roi_state.json"),
        "task": "seatbelt",
        "phone_model": False,
        "core": True,
    },
    "seatbelt_uncertain": {
        "queue": Path("datasets/manifests/v2_seatbelt_uncertain_ui_review.json"),
        "output": Path("datasets/manifests/pretrain_pending_approval/v2_seatbelt_uncertain.json"),
        "task": "uncertain",
        "phone_model": False,
        "core": False,
    },
    "adt_external": {
        "queue": Path("datasets/manifests/review_queue_adt_seatbelt_v1.json"),
        "output": Path("datasets/manifests/pretrain_pending_approval/v2_adt_external.json"),
        "task": "mixed",
        "phone_model": True,
        "core": False,
    },
    "hard_examples": {
        "queue": Path("datasets/manifests/v2_hard_review_queue.json"),
        "output": Path("datasets/manifests/pretrain_pending_approval/v2_hard_examples.json"),
        "task": "phone",
        "phone_model": True,
        "core": False,
    },
}


def _can_resume(output: Path, queue: Path, phone_model: Path | None) -> bool:
    if not output.is_file():
        return False
    try:
        report = json.loads(output.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return False
    if report.get("status") != "PASS" or report.get("reviewer_model") != REVIEWER_MODEL:
        return False
    if report.get("queue_sha256") != sha256_file(queue):
        return False
    embedded_model = report.get("phone_model", {})
    expected_model_hash = sha256_file(phone_model) if phone_model else None
    return embedded_model.get("sha256") == expected_model_hash


def prepare(
    selected: set[str],
    phone_model: Path,
    device: str,
    batch: int,
    force: bool,
    summary_path: Path,
) -> dict:
    unknown = selected - SPECS.keys()
    if unknown:
        raise ValueError(f"unknown review components: {sorted(unknown)}")
    reports = {}
    for name, spec in SPECS.items():
        if selected and name not in selected:
            continue
        queue = spec["queue"]
        output = spec["output"]
        model = phone_model if spec["phone_model"] else None
        if not force and _can_resume(output, queue, model):
            report = json.loads(output.read_text(encoding="utf-8"))
            report["resume_status"] = "REUSED_VERIFIED_REPORT"
        else:
            report = run(queue, output, spec["task"], model, device, batch)
            report["queue_sha256"] = sha256_file(queue)
            output.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
            report["resume_status"] = "GENERATED"
        reports[name] = {key: value for key, value in report.items() if key != "samples"}

    core = [name for name, spec in SPECS.items() if spec["core"] and name in reports]
    core_complete = len(core) == sum(spec["core"] for spec in SPECS.values())
    pretrain_review_complete = core_complete and all(
        reports[name]["status"] == "PASS" for name in core
    )
    uncertainty_rows = load_selected_uncertain_rows(
        SPECS["seatbelt_uncertain"]["output"],
        Path(
            "datasets/manifests/pretrain_pending_approval/"
            "v2_seatbelt_uncertain_visual_decisions.json"
        ),
    )
    uncertainty_counts = {
        split: sum(item.get("split") == split for item in uncertainty_rows)
        for split in MIN_UNCERTAIN_COUNTS
    }
    if "seatbelt_uncertain" in reports:
        base_eligible = int(reports["seatbelt_uncertain"].get("training_eligible", 0))
        reports["seatbelt_uncertain"]["source_heuristic_training_eligible"] = base_eligible
        reports["seatbelt_uncertain"]["training_eligible"] = len(uncertainty_rows)
        reports["seatbelt_uncertain"]["visual_semantic_overlay"] = {
            "path": (
                "datasets/manifests/pretrain_pending_approval/"
                "v2_seatbelt_uncertain_visual_decisions.json"
            ),
            "selected": len(uncertainty_rows),
            "counts_by_split": uncertainty_counts,
        }
    classifier_training_ready = all(
        uncertainty_counts[split] >= minimum for split, minimum in MIN_UNCERTAIN_COUNTS.items()
    )
    summary = {
        "schema_version": 1,
        "status": (
            "PASS" if all(value["status"] == "PASS" for value in reports.values()) else "FAIL"
        ),
        "reviewer_model": REVIEWER_MODEL,
        "components": reports,
        "readiness": {
            "pretrain_review_complete": pretrain_review_complete,
            "proposal_detector_training_ready": pretrain_review_complete
            and all(reports[name]["training_eligible"] > 0 for name in core),
            "classifier_training_ready": classifier_training_ready,
            "uncertain_training_eligible_by_split": uncertainty_counts,
            "minimum_uncertain_by_split": MIN_UNCERTAIN_COUNTS,
            "exploratory_training_ready": pretrain_review_complete
            and classifier_training_ready
            and all(reports[name]["training_eligible"] > 0 for name in core),
            "human_verified": False,
            "governed_training_ready": False,
        },
        "policy": (
            "Exploratory training may consume only model-assisted PASS samples with explicit "
            "automated-review provenance. Human confirmation remains a separate append-only "
            "governance layer."
        ),
    }
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run or resume every V2 model-assisted review queue"
    )
    parser.add_argument("--only", nargs="*", choices=sorted(SPECS), default=[])
    parser.add_argument("--phone-model", type=Path, default=Path("models/auxiliary/yolo11n.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--force", action="store_true")
    parser.add_argument(
        "--summary", type=Path, default=Path("reports/v2_pretrain_review_summary.json")
    )
    args = parser.parse_args()
    summary = prepare(
        set(args.only), args.phone_model, args.device, args.batch, args.force, args.summary
    )
    print(json.dumps(summary, indent=2))
    raise SystemExit(0 if summary["status"] == "PASS" else 1)


if __name__ == "__main__":
    main()
