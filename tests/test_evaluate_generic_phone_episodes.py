import json
from pathlib import Path

from training.dmd.evaluate_generic_phone_episodes import evaluate_run


def test_generic_phone_episode_diagnostic_collapses_adjacent_actions(tmp_path: Path) -> None:
    run = tmp_path / "run"
    subject = run / "subjects" / "gC-14"
    subject.mkdir(parents=True)
    first = {"start_seconds": 10.0, "end_seconds": 11.0, "source_action": "texting"}
    second = {"start_seconds": 11.1, "end_seconds": 20.0, "source_action": "phonecall"}
    metrics = {
        "matched_events": [{"ground_truth": first}],
        "unmatched_gt": [second],
        "evaluated_minutes": 1.0,
    }
    (subject / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    prediction = {"start_seconds": 9.9, "end_seconds": 20.1, "event_type": "PHONE"}
    (subject / "event_predictions.jsonl").write_text(
        json.dumps(prediction) + "\n", encoding="utf-8"
    )
    (run / "event_matches.json").write_text(
        json.dumps({"cohort": ["gC-14"]}), encoding="utf-8"
    )

    result = evaluate_run(run, gap_seconds=3.0)

    assert result["primary_v2_metric_replacement"] is False
    assert result["total_gt_events"] == 1
    assert result["true_positives"] == 1
    assert result["false_positives"] == 0
    assert result["false_negatives"] == 0
    assert result["f1"] == 1.0
