from pathlib import Path

from training.dmd.replay_phone_tracklets import _read_jsonl


def test_read_jsonl_ignores_blank_lines(tmp_path: Path) -> None:
    path = tmp_path / "rows.jsonl"
    path.write_text('{"timestamp": 1.0}\n\n{"timestamp": 2.0}\n', encoding="utf-8")

    assert _read_jsonl(path) == [{"timestamp": 1.0}, {"timestamp": 2.0}]
