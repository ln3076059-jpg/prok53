"""Tests for Holdout Access Guard."""
import json
import pytest
from pathlib import Path
from training.dmd.holdout_guard import (
    HoldoutAccessError,
    assert_holdout_untouched,
    assert_not_consumed_holdout_for_development,
    is_holdout_subject,
    is_consumed_holdout,
    is_holdout_unlocked,
)


def test_holdout_detection():
    assert is_holdout_subject("gE-28")
    assert is_holdout_subject("ge_28")
    assert is_holdout_subject("gZ-37")
    assert is_holdout_subject("gz_37")
    assert is_holdout_subject("datasets/external_dmd/holdout/gE-28/archive.tar.gz")
    assert not is_holdout_subject("gZ-36")
    assert not is_holdout_subject("gB-9")
    assert not is_holdout_subject("gC-14")


def test_consumed_holdout_detection():
    assert is_consumed_holdout("gZ-37")
    assert is_consumed_holdout("gE-28")
    assert not is_consumed_holdout("gC-14")
    assert not is_consumed_holdout("gZ-36")
    assert not is_consumed_holdout("gB-9")


def test_assert_holdout_untouched_raises_before_freeze(tmp_path, monkeypatch):
    fake_freeze = tmp_path / "non_existent_freeze.json"
    monkeypatch.setattr("training.dmd.holdout_guard.FREEZE_FILE_PATH", fake_freeze)
    with pytest.raises(HoldoutAccessError) as exc_info:
        assert_holdout_untouched("gE-28", caller_action="decode_frame")
    assert "HOLDOUT ISOLATION BREACH PREVENTED" in str(exc_info.value)
    assert "decode_frame" in str(exc_info.value)


def test_consumed_holdouts_blocked_for_development():
    for holdout in ["gZ-37", "gE-28"]:
        for action in ["training", "tuning", "ablation", "calibration", "hard_negative_mining"]:
            with pytest.raises(HoldoutAccessError) as exc_info:
                assert_not_consumed_holdout_for_development(holdout, caller_action=action)
            assert "CONSUMED HOLDOUT BREACH PREVENTED" in str(exc_info.value)

            with pytest.raises(HoldoutAccessError):
                assert_holdout_untouched(holdout, caller_action=action)


def test_assert_development_subjects_pass():
    # Development pool subjects should pass without error
    assert_holdout_untouched("gZ-36", caller_action="extract")
    assert_holdout_untouched("gB-9", caller_action="extract")
    assert_holdout_untouched("gC-14", caller_action="extract")
    assert_holdout_untouched("gZ-36", caller_action="ablation")
    assert_holdout_untouched("gB-9", caller_action="tuning")


def test_unlocked_when_freeze_authorized(tmp_path, monkeypatch):
    fake_freeze = tmp_path / "freeze.json"
    fake_freeze.write_text(
        json.dumps({"GE28_HOLDOUT_UNLOCK": "AUTHORIZED_FOR_ONE_SHOT_EVALUATION"}),
        encoding="utf-8"
    )
    assert is_holdout_unlocked(fake_freeze)
