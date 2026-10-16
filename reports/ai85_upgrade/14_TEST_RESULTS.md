# Executed software checks

```json
{
  "ruff": {
    "command": [
      "C:\\Users\\TSC\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "ruff",
      "check",
      "training/dmd/ai85_action.py",
      "training/dmd/ai85_audit.py",
      "training/dmd/ai85_detector_probe.py",
      "training/dmd/ai85_native_embeddings.py",
      "training/dmd/run_ai85_upgrade.py",
      "training/dmd/render_ai85_reports.py",
      "training/dmd/verify_ai85.py",
      "tests/test_ai85_upgrade.py"
    ],
    "exit_code": 0,
    "status": "PASS",
    "seconds": 2.284093700000085,
    "log": "reports/ai85_upgrade/ruff.log"
  },
  "compile": {
    "command": [
      "C:\\Users\\TSC\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "compileall",
      "-q",
      "training/dmd/ai85_action.py",
      "training/dmd/ai85_audit.py",
      "training/dmd/ai85_detector_probe.py",
      "training/dmd/ai85_native_embeddings.py",
      "training/dmd/run_ai85_upgrade.py",
      "training/dmd/render_ai85_reports.py",
      "training/dmd/verify_ai85.py",
      "tests/test_ai85_upgrade.py"
    ],
    "exit_code": 0,
    "status": "PASS",
    "seconds": 0.4191798000001654,
    "log": "reports/ai85_upgrade/compile.log"
  },
  "pytest_full": {
    "command": [
      "C:\\Users\\TSC\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "tests",
      "-q",
      "--continue-on-collection-errors",
      "--junitxml",
      "reports\\ai85_upgrade\\pytest_full.xml"
    ],
    "exit_code": 1,
    "status": "FAIL",
    "seconds": 74.65459230000033,
    "log": "reports/ai85_upgrade/pytest_full.log",
    "counts": {
      "tests": 537,
      "errors": 74,
      "failures": 37,
      "skipped": 1,
      "passed": 425
    }
  },
  "pytest_targeted": {
    "command": [
      "C:\\Users\\TSC\\AppData\\Local\\Programs\\Python\\Python311\\python.exe",
      "-m",
      "pytest",
      "-q",
      "tests/test_ai85_upgrade.py",
      "tests/test_v7_action_models.py",
      "tests/test_v7_result_integrity.py",
      "tests/test_v6_action_hierarchy.py",
      "tests/test_v6_seatbelt_evaluation.py",
      "tests/test_v5_video_action.py",
      "tests/test_holdout_guard.py",
      "tests/test_engineering_freeze.py",
      "tests/test_runtime_v2.py",
      "tests/test_v4_experiment_runner.py",
      "--junitxml",
      "reports\\ai85_upgrade\\pytest_targeted.xml"
    ],
    "exit_code": 0,
    "status": "PASS",
    "seconds": 26.574944200000118,
    "log": "reports/ai85_upgrade/pytest_targeted.log",
    "counts": {
      "tests": 89,
      "errors": 0,
      "failures": 0,
      "skipped": 0,
      "passed": 89
    }
  }
}
```
```json
{
  "missing_legacy_artifact_groups": {
    "v2_event_sequence_annotation.schema.json": 24,
    "v2_review1.schema.json": 31,
    "v2_external_test_policy.yaml": 56
  },
  "all_111_failures_or_errors_reference_missing_legacy_artifacts": true,
  "restored_or_modified_deleted_user_data": false
}
```

Logs and JUnit XML are retained. `test_results.json` contains all failure details. The full suite is reported with its actual status, including failures from deleted legacy `datasets/` paths. Tests were not skipped or rewritten to hide those failures. The targeted suite includes the new regression checks, V5–V7 compatibility, holdout protection, runtime configuration, backend queue/integration and experiment runners.
