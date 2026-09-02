from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from training.apply_review_decisions import apply as apply_human_decisions
from training.apply_review_decisions import resolve_ingested_image, yolo_line
from training.common import sha256_file

REVIEW1_TRAINABLE = {
    "REVIEW1_APPROVED_UNDER_ADMIN_DELEGATION",
    "REVIEW1_ACCEPTED_PROPOSAL",
    "REVIEW1_CORRECTION_PROPOSAL",
}
HUMAN_APPROVED = {"APPROVED", "APPROVED_NEGATIVE"}


def read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def decision_lanes(history: list[dict]) -> tuple[dict[str, dict], dict[str, dict]]:
    review1: dict[str, dict] = {}
    human: dict[str, dict] = {}
    for item in history:
        sample_id = str(item["sample_id"])
        if item.get("reviewer_id") == "review1" and item.get("reviewer_type") == "AI":
            review1[sample_id] = item
        if item.get("reviewer_type") == "HUMAN" and item.get("status") in HUMAN_APPROVED:
            human[sample_id] = item
    return review1, human


def _verify_source(record: dict, decision: dict) -> Path:
    image = resolve_ingested_image(record)
    actual = sha256_file(image).lower()
    expected = str(decision.get("source_sha256", "")).lower()
    if not expected or actual != expected:
        raise ValueError(f"SOURCE_HASH_MISMATCH for {record['sample_id']}")
    return image


def materialize(
    records: list[dict],
    history: list[dict],
    bootstrap_output: Path,
    governed_output: Path,
) -> tuple[list[dict], list[dict]]:
    review1, human = decision_lanes(history)
    bootstrap_output.mkdir(parents=True, exist_ok=True)
    bootstrap_manifest: list[dict] = []
    record_by_id = {
        str(item["sample_id"]): item for item in records if item.get("sample_id")
    }

    for sample_id, decision in sorted(review1.items()):
        record = record_by_id.get(sample_id)
        if record is None:
            continue
        source = _verify_source(record, decision)
        status = decision.get("new_status") or decision.get("status")
        annotations = decision.get("reviewed_annotations", decision.get("annotations", []))
        trainable = status in REVIEW1_TRAINABLE
        item = {
            **record,
            "review_status": status,
            "reviewer_id": "review1",
            "reviewer_type": "AI",
            "delegated_by": "admin",
            "approval_authority_id": "admin",
            "decision_id": decision.get("decision_id"),
            "reviewed_annotations": annotations,
            "lane": "v2_review1_reviewed_bootstrap",
            "usage": "MODEL_ASSISTED",
            "delegation_status": "ADMIN_DELEGATED",
            "human_approval_status": "NOT_HUMAN_APPROVED",
            "governance_status": "NOT_GOVERNED",
            "governance_eligible": False,
            "training_eligible": trainable,
        }
        if trainable:
            image_target = bootstrap_output / f"{sample_id}{source.suffix.lower()}"
            label_target = bootstrap_output / f"{sample_id}.txt"
            if image_target.exists() or label_target.exists():
                raise FileExistsError(f"Review 1 output already exists for {sample_id}")
            shutil.copy2(source, image_target)
            labels = "\n".join(yolo_line(annotation) for annotation in annotations)
            label_target.write_text(labels + ("\n" if labels else ""), encoding="utf-8")
            item["reviewed_path"] = str(image_target.resolve())
            item["label_path"] = str(label_target.resolve())
        bootstrap_manifest.append(item)

    governed_records = []
    governed_decisions = {}
    for sample_id, decision in human.items():
        record = record_by_id.get(sample_id)
        if record is None:
            continue
        _verify_source(record, decision)
        governed_records.append(record)
        governed_decisions[sample_id] = decision
    governed_manifest = apply_human_decisions(
        governed_records,
        governed_decisions,
        governed_output,
    )
    for item in governed_manifest:
        item.update(
            {
                "lane": "v2_governed",
                "governance_status": "GOVERNED",
                "governance_eligible": True,
            }
        )
    return bootstrap_manifest, governed_manifest


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(json.dumps(item, sort_keys=True) for item in records)
    path.write_text(body + ("\n" if body else ""), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Materialize separate Review 1 and governed lanes")
    parser.add_argument("--ingest-manifest", type=Path, action="append", required=True)
    parser.add_argument(
        "--decisions",
        type=Path,
        default=Path("datasets/manifests/review_decisions.jsonl"),
    )
    parser.add_argument(
        "--bootstrap-output", type=Path, default=Path("datasets/v2_review1_reviewed_bootstrap")
    )
    parser.add_argument("--governed-output", type=Path, default=Path("datasets/v2_governed"))
    parser.add_argument(
        "--bootstrap-manifest",
        type=Path,
        default=Path("datasets/manifests/v2_review1_reviewed_bootstrap.jsonl"),
    )
    parser.add_argument(
        "--governed-manifest",
        type=Path,
        default=Path("datasets/manifests/v2_governed.jsonl"),
    )
    args = parser.parse_args()
    records = [item for path in args.ingest_manifest for item in read_jsonl(path)]
    bootstrap, governed = materialize(
        records,
        read_jsonl(args.decisions),
        args.bootstrap_output,
        args.governed_output,
    )
    write_jsonl(args.bootstrap_manifest, bootstrap)
    write_jsonl(args.governed_manifest, governed)
    print(
        json.dumps(
            {
                "review1_lane_records": len(bootstrap),
                "governed_lane_records": len(governed),
                "governed_ready": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
