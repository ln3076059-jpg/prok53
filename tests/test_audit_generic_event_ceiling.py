import json
from pathlib import Path

from training.dmd.audit_generic_event_ceiling import (
    audit_generic_event_ceiling,
    cluster_generic_episodes,
)


def _event(start: float, end: float, action: str) -> dict:
    return {
        "start_seconds": start,
        "end_seconds": end,
        "source_action": action,
    }


def test_cluster_generic_episodes_merges_action_transitions_inside_gap() -> None:
    events = [
        _event(10.0, 11.0, "texting"),
        _event(11.1, 20.0, "phonecall"),
        _event(24.0, 25.0, "texting"),
    ]

    clusters = cluster_generic_episodes(events, gap_seconds=3.0)

    assert [len(cluster) for cluster in clusters] == [2, 1]


def test_audit_reports_architecture_consistent_ceiling(tmp_path: Path) -> None:
    subject = tmp_path / "gC-14"
    subject.mkdir()
    metrics = {
        "matched_events": [
            {"ground_truth": _event(10.0, 11.0, "texting")},
            {"ground_truth": _event(11.1, 20.0, "phonecall")},
        ],
        "unmatched_gt": [_event(30.0, 31.0, "texting")],
    }
    path = subject / "metrics.json"
    path.write_text(json.dumps(metrics), encoding="utf-8")

    result = audit_generic_event_ceiling([path], gap_seconds=3.0)

    assert result["action_ground_truth_count"] == 3
    assert result["generic_episode_count"] == 2
    assert result["architecture_consistent_recall_ceiling"] == 0.666667
    assert result["target_recall_above_0_85_representable"] is False
    assert result["per_subject"]["gC-14"]["merged_clusters"][0]["action_count"] == 2
