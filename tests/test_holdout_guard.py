"""Tests for Holdout Access Guard."""
import json
import pytest
from pathlib import Path
from training.dmd.holdout_guard import (
    HoldoutAccessError,
    assert_holdout_untouched,
    is_holdout_subject,
    is_holdout_unlocked,
)


def test_holdout_detection():
    assert is_holdout_subject("gE-28")
    assert is_holdout_subject("ge_28")
    assert is_holdout_subject("datasets/external_dmd/holdout/gE-28/archive.tar.gz")
    assert not is_holdout_subject("gZ-36")
    assert not is_holdout_subject("gB-9")
    assert not is_holdout_subject("gC-14")


def test_assert_holdout_untouched_raises_before_freeze(tmp_path, monkeypatch):
    fake_freeze = tmp_path / "non_existent_freeze.json"
    monkeypatch.setattr("training.dmd.holdout_guard.FREEZE_FILE_PATH", fake_freeze)
    with pytest.raises(HoldoutAccessError) as exc_info:
        assert_holdout_untouched("gE-28", caller_action="decode_frame")
    assert "HOLDOUT ISOLATION BREACH PREVENTED" in str(exc_info.value)
    assert "decode_frame" in str(exc_info.value)


def test_assert_development_subjects_pass():
    # Development pool subjects should pass without error
    assert_holdout_untouched("gZ-36", caller_action="extract")
    assert_holdout_untouched("gB-9", caller_action="extract")
    assert_holdout_untouched("gC-14", caller_action="extract")


def test_unlocked_when_freeze_authorized(tmp_path, monkeypatch):
    fake_freeze = tmp_path / "freeze.json"
    fake_freeze.write_text(
        json.dumps({"GE28_HOLDOUT_UNLOCK": "AUTHORIZED_FOR_ONE_SHOT_EVALUATION"}),
        encoding="utf-8"
    )
    assert is_holdout_unlocked(fake_freeze)
