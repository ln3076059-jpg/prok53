import json
from pathlib import Path

import pytest

from training.dmd.merge_subject_runs import JSONL_FILES, merge_subject_runs


def _write_run(
    root: Path,
    subject_id: str,
    *,
    tp: int,
    fp: int,
    fn: int,
    mode: str = "D3",
) -> None:
    subject = root / "subjects" / subject_id
    subject.mkdir(parents=True)
    gt = tp + fn
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / gt if gt else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    metrics = {
        "true_positives": tp,
        "false_positives": fp,
        "false_negatives": fn,
        "total_gt_events": gt,
        "evaluated_minutes": 5.0,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "false_alarms_per_minute": round(fp / 5.0, 4),
        "effective_fps": 3.0,
        "action_provenance": None,
    }
    (subject / "metrics.json").write_text(json.dumps(metrics), encoding="utf-8")
    for filename in JSONL_FILES:
        (subject / filename).write_text(
            json.dumps({"subject_id": subject_id}) + "\n", encoding="utf-8"
        )
    aggregate = {
        "schema_version": "v21.0",
        "cohort": [subject_id],
        "detector_mode": mode,
        "detector_imgsz": 640,
        "adaptive_sampling": False,
        "adaptive_stride": 3,
        "adaptive_hold_seconds": 2.0,
        "action_mode": "NONE",
        "action_causal_mode": "TARGET_ALIGNED",
        "missing_checkpoint_policy": "ERROR",
        "device": "cpu",
        "force_handheld_context": False,
        "verified_association": True,
        "action_gating": False,
    }
    (root / "event_matches.json").write_text(json.dumps(aggregate), encoding="utf-8")


def test_merge_subject_runs_recomputes_micro_metrics(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    output = tmp_path / "merged"
    _write_run(first, "gC-14", tp=8, fp=2, fn=1)
    _write_run(second, "gZ-36", tp=9, fp=1, fn=2)

    merged = merge_subject_runs([first, second], output)

    assert merged["cohort"] == ["gC-14", "gZ-36"]
    assert merged["true_positives"] == 17
    assert merged["false_positives"] == 3
    assert merged["false_negatives"] == 3
    assert merged["total_gt_events"] == 20
    assert merged["precision"] == 0.85
    assert merged["recall"] == 0.85
    assert merged["f1"] == 0.85
    rows = (output / "autonomous_predictions.jsonl").read_text(
        encoding="utf-8"
    ).splitlines()
    assert [json.loads(row)["subject_id"] for row in rows] == ["gC-14", "gZ-36"]


def test_merge_subject_runs_rejects_mixed_configuration(tmp_path: Path) -> None:
    first = tmp_path / "first"
    second = tmp_path / "second"
    _write_run(first, "gC-14", tp=1, fp=0, fn=0, mode="D3")
    _write_run(second, "gZ-36", tp=1, fp=0, fn=0, mode="D5")

    with pytest.raises(ValueError, match="Incompatible run configurations"):
        merge_subject_runs([first, second], tmp_path / "merged")
