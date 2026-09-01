from __future__ import annotations

import csv

from training.calibrate_thresholds import calibrate
from training.common import sha256_file


def test_threshold_calibration_records_model_and_validation_provenance(tmp_path):
    scores = tmp_path / "scores.csv"
    with scores.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["split", "class_name", "y_true", "score"])
        writer.writeheader()
        writer.writerows(
            [
                {"split": "val", "class_name": "phone", "y_true": 1, "score": 0.9},
                {"split": "val", "class_name": "phone", "y_true": 0, "score": 0.2},
                {"split": "test", "class_name": "phone", "y_true": 1, "score": 0.8},
            ]
        )
    model = tmp_path / "phone.pt"
    manifest = tmp_path / "validation.jsonl"
    model.write_bytes(b"real-model-placeholder-for-contract-test")
    manifest.write_text('{"sample_id":"validation-1"}\n', encoding="utf-8")

    report = calibrate(scores, tmp_path / "calibration.json", model, manifest)

    assert report["status"] == "CALIBRATED_ON_VALIDATION"
    assert report["model"]["sha256"] == sha256_file(model)
    assert report["validation_dataset_manifest"]["sha256"] == sha256_file(manifest)
    assert report["input_split_rows"] == {"test": 1, "val": 2}
    assert report["test_rows_used"] == 0
