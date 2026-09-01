from __future__ import annotations

import copy
import json
from collections import Counter
from pathlib import Path

from training.common import sha256_file

VISUAL_REVIEWER_MODEL = "roadwatch_visual_semantic_review_v2"
VISUAL_EVIDENCE_LEVEL = "DIRECT_MODEL_ASSISTED_VISUAL_UNOBSERVABLE_BELT_PATH"


def load_selected_uncertain_rows(review_path: Path, decisions_path: Path) -> list[dict]:
    """Merge an auditable visual-decision overlay into the uncertainty queue.

    The source review report stays immutable. The overlay is accepted only while
    its recorded source hash matches, so regenerating the queue cannot silently
    reuse stale semantic decisions.
    """

    review = json.loads(review_path.read_text(encoding="utf-8"))
    decisions = json.loads(decisions_path.read_text(encoding="utf-8"))
    if review.get("status") != "PASS":
        raise ValueError(f"uncertainty source review did not pass: {review_path}")
    if decisions.get("status") != "PASS":
        raise ValueError(f"uncertainty visual decisions did not pass: {decisions_path}")
    if decisions.get("source_review_sha256") != sha256_file(review_path):
        raise ValueError("uncertainty visual decisions are stale for the source review")

    selected = {str(item["candidate_id"]): item for item in decisions["decisions"]}
    if len(selected) != len(decisions["decisions"]):
        raise ValueError("duplicate candidate_id in uncertainty visual decisions")

    merged = []
    seen = set()
    for source_item in review["samples"]:
        candidate_id = str(source_item["candidate_id"])
        decision = selected.get(candidate_id)
        if decision is None:
            continue
        if decision["sample_id"] != source_item["sample_id"]:
            raise ValueError(f"sample mismatch for uncertainty candidate: {candidate_id}")
        if decision["split"] != source_item["split"]:
            raise ValueError(f"split mismatch for uncertainty candidate: {candidate_id}")
        item = copy.deepcopy(source_item)
        proposal = item["proposal_review"]
        proposal.update(
            {
                "status": "MODEL_ASSISTED_VISUAL_DECISION_PENDING_APPROVAL",
                "reviewer_type": "AUTOMATED",
                "reviewer_model": VISUAL_REVIEWER_MODEL,
                "decision": "UNCERTAIN",
                "tier": "VISUALLY_CONFIRMED_UNCERTAIN",
                "confidence": 0.95,
                "reason": list(decision["reason"]),
                "evidence_level": VISUAL_EVIDENCE_LEVEL,
                "training_eligible": True,
                "training_target": "uncertain_or_occluded",
                "governance_eligible": False,
                "requires_human_confirmation": True,
                "visual_decision_id": decision["decision_id"],
                "visual_decision_sha256": decision["decision_sha256"],
            }
        )
        merged.append(item)
        seen.add(candidate_id)
    missing = sorted(set(selected) - seen)
    if missing:
        raise ValueError(f"visual decisions absent from source review: {missing[:5]}")

    actual_counts = Counter(str(item["split"]) for item in merged)
    recorded_counts = decisions.get("counts_by_split", {})
    if any(
        actual_counts[split] != int(recorded_counts.get(split, -1))
        for split in recorded_counts
    ):
        raise ValueError("uncertainty visual-decision split counts are stale")
    return merged
