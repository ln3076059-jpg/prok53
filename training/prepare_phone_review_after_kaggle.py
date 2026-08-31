from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.import_phone_bootstrap_recovery import install_recovery
from training.merge_phone_proposals import merge
from training.propose_phone_boxes import load_records, propose


def main() -> None:
    parser = argparse.ArgumentParser(description="Import Kaggle phone weight and prepare Mendeley phone review")
    parser.add_argument("recovery_archive", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--confidence", type=float, default=0.1)
    parser.add_argument("--image-size", type=int, default=960)
    parser.add_argument("--chunk-size", type=int, default=16)
    parser.add_argument(
        "--ingest", type=Path, default=Path("datasets/manifests/ingest_mendeley_driver_risk_v1.jsonl")
    )
    parser.add_argument(
        "--queue", type=Path, default=Path("datasets/manifests/review_queue_mendeley_driver_risk_v1.json")
    )
    parser.add_argument(
        "--proposals", type=Path, default=Path("datasets/manifests/phone_proposals_mendeley_bootstrap_001.json")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("datasets/manifests/review_queue_mendeley_phone_bootstrap_001.json")
    )
    args = parser.parse_args()

    imported = install_recovery(args.recovery_archive, Path("models/proposals/phone_bootstrap_001"))
    proposal_report = propose(
        load_records(args.ingest, "mendeley_driver_risk_behavior_v1"),
        source_class_id=3,
        model_name=imported["best_weights_path"],
        confidence=args.confidence,
        image_size=args.image_size,
        padding=0.2,
        chunk_size=args.chunk_size,
        device=args.device,
    )
    args.proposals.parent.mkdir(parents=True, exist_ok=True)
    args.proposals.write_text(json.dumps(proposal_report, indent=2, sort_keys=True), encoding="utf-8")
    queue = json.loads(args.queue.read_text(encoding="utf-8"))
    review = merge(queue, proposal_report)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(review, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"import": imported, "proposals": proposal_report["summary"], "review_queue": str(args.output), "review_samples": len(review)}, indent=2))


if __name__ == "__main__":
    main()
