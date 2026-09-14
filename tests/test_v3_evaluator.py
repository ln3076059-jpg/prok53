"""Tests for V3 Evaluator."""
import pytest
from pathlib import Path
from training.dmd.v3_evaluator import calculate_per_action_recall, V3AblationResult
from training.dmd.holdout_guard import HoldoutAccessError
from training.dmd.v3_evaluator import run_v3_evaluation_on_subject


def test_evaluator_blocks_ge28_before_freeze(tmp_path):
    """Calling run_v3_evaluation_on_subject on gE-28 before freeze must raise HoldoutAccessError."""
    with pytest.raises(HoldoutAccessError):
        run_v3_evaluation_on_subject(
            subject_dir=tmp_path / "gE-28",
            subject_id="gE-28",
            detector=None,
        )


def test_calculate_per_action_recall():
    gt_intervals = [
        {"source_action": "driver_actions/phonecall_right"},
        {"source_action": "driver_actions/phonecall_right"},
        {"source_action": "driver_actions/phonecall_left"},
        {"source_action": "driver_actions/texting_right"},
    ]
    matched_events = [
        {"gt": {"source_action": "driver_actions/phonecall_right"}},
        {"gt": {"source_action": "driver_actions/texting_right"}},
    ]
    recalls = calculate_per_action_recall(gt_intervals, matched_events)
    assert recalls["phonecall_right"] == 0.50
    assert recalls["phonecall_left"] == 0.0
    assert recalls["texting_right"] == 1.0
