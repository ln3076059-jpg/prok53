"""Audit the semantic ceiling of generic PHONE events against action GT.

The V21 event stream emits one generic PHONE class, while DMD annotations split
continuous phone use into texting and phone-call actions.  Ground-truth actions
separated by no more than the event engine's gap/cooldown cannot be faithfully
represented as distinct generic episodes without artificial fragmentation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Iterable


def _ground_truth(metrics: dict[str, Any]) -> list[dict[str, Any]]:
    events = [row["ground_truth"] for row in metrics["matched_events"]]
    events.extend(metrics["unmatched_gt"])
    return sorted(events, key=lambda row: float(row["start_seconds"]))


def cluster_generic_episodes(
    events: Iterable[dict[str, Any]], gap_seconds: float
) -> list[list[dict[str, Any]]]:
    if gap_seconds < 0:
        raise ValueError("gap_seconds must be non-negative")
    clusters: list[list[dict[str, Any]]] = []
    for event in sorted(events, key=lambda row: float(row["start_seconds"])):
        if not clusters:
            clusters.append([event])
            continue
        previous = clusters[-1][-1]
        gap = float(event["start_seconds"]) - float(previous["end_seconds"])
        if gap <= gap_seconds:
            clusters[-1].append(event)
        else:
            clusters.append([event])
    return clusters


def audit_generic_event_ceiling(
    subject_metric_paths: Iterable[Path], gap_seconds: float
) -> dict[str, Any]:
    per_subject: dict[str, Any] = {}
    total_gt = 0
    total_episodes = 0
    for path in subject_metric_paths:
        metrics = json.loads(Path(path).read_text(encoding="utf-8"))
        events = _ground_truth(metrics)
        if not events:
            raise ValueError(f"No ground-truth events in {path}")
        subject_id = Path(path).parent.name
        if subject_id in per_subject:
            raise ValueError(f"Duplicate subject: {subject_id}")
        clusters = cluster_generic_episodes(events, gap_seconds)
        total_gt += len(events)
        total_episodes += len(clusters)
        per_subject[subject_id] = {
            "action_ground_truth_count": len(events),
            "generic_episode_count": len(clusters),
            "merged_clusters": [
                {
                    "start_seconds": cluster[0]["start_seconds"],
                    "end_seconds": cluster[-1]["end_seconds"],
                    "action_count": len(cluster),
                    "actions": [
                        {
                            "start_seconds": event["start_seconds"],
                            "end_seconds": event["end_seconds"],
                            "source_action": event["source_action"],
                        }
                        for event in cluster
                    ],
                }
                for cluster in clusters
                if len(cluster) > 1
            ],
        }

    architecture_consistent_recall_ceiling = (
        total_episodes / total_gt if total_gt else 0.0
    )
    perfect_precision_f1_at_ceiling = (
        2 * architecture_consistent_recall_ceiling
        / (1 + architecture_consistent_recall_ceiling)
        if architecture_consistent_recall_ceiling
        else 0.0
    )
    return {
        "schema_version": "1.0.0",
        "audit_type": "GENERIC_PHONE_VS_ACTION_GT_SEMANTIC_CEILING",
        "gap_seconds": gap_seconds,
        "output_semantics": "ONE_GENERIC_PHONE_EVENT_PER_CONTINUOUS_PHONE_USE_EPISODE",
        "action_ground_truth_count": total_gt,
        "generic_episode_count": total_episodes,
        "architecture_consistent_recall_ceiling": round(
            architecture_consistent_recall_ceiling, 6
        ),
        "perfect_precision_f1_at_recall_ceiling": round(
            perfect_precision_f1_at_ceiling, 6
        ),
        "target_recall_above_0_85_representable": (
            architecture_consistent_recall_ceiling > 0.85
        ),
        "interpretation": (
            "This is an architecture/label-semantics ceiling, not a mathematical "
            "ceiling of the matcher. Artificially fragmenting generic PHONE output "
            "could produce more predictions but would not represent distinct actions."
        ),
        "required_resolution": [
            "Add causal action segmentation (for example TEXTING vs PHONE_CALL), or",
            "Declare a separate generic-phone episode benchmark before evaluation.",
        ],
        "per_subject": per_subject,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--subject-metrics", action="append", required=True, type=Path)
    parser.add_argument("--gap-seconds", type=float, default=3.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    result = audit_generic_event_ceiling(args.subject_metrics, args.gap_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
