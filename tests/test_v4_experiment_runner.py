import json

from training.dmd.run_v4_experiments import DEVELOPMENT_SUBJECTS, preflight


def test_v4_preflight_is_development_only_and_hashes_inputs(tmp_path, monkeypatch):
    derivatives = tmp_path / "derivatives"
    derivatives.mkdir()
    record = {
        "timestamp_seconds": 0.0,
        "frame_index": 0,
        "body_phone_conf": 0.1,
        "face_phone_conf": 0.0,
        "hands_phone_conf": 0.0,
        "body_available": True,
        "face_available": True,
        "hands_available": True,
    }
    for subject_id in DEVELOPMENT_SUBJECTS:
        (derivatives / f"v3_features_{subject_id}.json").write_text(
            json.dumps({"records": [record]}), encoding="utf-8"
        )
    config = tmp_path / "v4.yaml"
    config.write_text("model_version: test\n", encoding="utf-8")
    monkeypatch.setattr("training.dmd.run_v4_experiments._git_head", lambda: "a" * 40)

    result = preflight(derivatives, config)

    assert result["status"] == "READY"
    assert result["dataset_role"] == "DEVELOPMENT_ONLY"
    assert result["consumed_holdouts_forbidden"] == ["gZ-37", "gE-28"]
    assert len(result["inputs"]) == 3
    assert len(result["input_set_sha256"]) == 64


def test_v4_preflight_reports_missing_inputs_without_using_holdout(tmp_path, monkeypatch):
    config = tmp_path / "v4.yaml"
    config.write_text("model_version: test\n", encoding="utf-8")
    monkeypatch.setattr("training.dmd.run_v4_experiments._git_head", lambda: "b" * 40)

    result = preflight(tmp_path / "missing", config)

    assert result["status"] == "BLOCKED"
    assert len(result["blockers"]) == 3
    assert all("gZ-37" not in item and "gE-28" not in item for item in result["blockers"])
