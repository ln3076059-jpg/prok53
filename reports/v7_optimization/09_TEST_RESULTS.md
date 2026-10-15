# Test results

Machine-readable results: `v7_test_results.json`.

```json
{
  "schema_version": 1,
  "status": "COMPLETED",
  "python_compilation": {
    "status": "PASS",
    "compiler": "Python 3.11.9 py_compile",
    "verified_paths": [
      "training/dmd/v7_action_models.py",
      "training/dmd/run_v7_action_fold.py",
      "training/dmd/run_v7_experiments.py",
      "training/dmd/analyze_v7_safe_failures.py",
      "training/dmd/render_v7_reports.py",
      "tests/test_v7_action_models.py",
      "tests/test_v7_result_integrity.py"
    ]
  },
  "ruff_gate": {
    "status": "PASS",
    "tool": "ruff 0.8.4",
    "checked_targets": [
      "training/dmd/v7_action_models.py",
      "training/dmd/run_v7_action_fold.py",
      "training/dmd/run_v7_experiments.py",
      "training/dmd/analyze_v7_safe_failures.py",
      "training/dmd/render_v7_reports.py",
      "tests/test_v7_action_models.py",
      "tests/test_v7_result_integrity.py"
    ],
    "violations": 0
  },
  "targeted_test_suites": {
    "test_v7_action_models": {
      "status": "PASS",
      "passed": 3,
      "failed": 0,
      "scope": "Binary gate probabilities normalization, head shapes, independent hierarchical objectives"
    },
    "test_v7_result_integrity": {
      "status": "PASS",
      "passed": 5,
      "failed": 0,
      "scope": "Subject-disjoint LOSO, holdout isolation, metric reconciliation, gate enforcement, checkpoint existence"
    },
    "test_v6_action_hierarchy": {
      "status": "PASS",
      "passed": 4,
      "failed": 0,
      "scope": "Hierarchical head outputs, probabilities normalization, sqrt-inverse weights, coarse targets"
    },
    "test_v6_seatbelt_evaluation": {
      "status": "PASS",
      "passed": 2,
      "failed": 0,
      "scope": "Seatbelt raw metrics reconciliation, calibration audit"
    },
    "test_v5_video_action": {
      "status": "PASS",
      "passed": 4,
      "failed": 0,
      "scope": "Event extraction, hysteresis, release ratio, evidence fusion"
    },
    "test_v5_result_integrity": {
      "status": "PASS",
      "passed": 7,
      "failed": 0,
      "scope": "Historical V5 LOSO verification, gate verification, matrix totals"
    },
    "test_holdout_guard": {
      "status": "PASS",
      "passed": 6,
      "failed": 0,
      "scope": "Protection of consumed holdouts gZ-37 and gE-28"
    },
    "test_common": {
      "status": "PASS",
      "passed": 1,
      "failed": 0,
      "scope": "File sha256 and json hashing determinism"
    },
    "summary": {
      "total_passed": 32,
      "total_failed": 0
    }
  },
  "external_and_dataset_failures_classification": {
    "status": "EXPLAINED_DATASET_ARTIFACT_BLOCKER",
    "code_regression": false,
    "cause": "Missing local dataset bootstrap and schema files in datasets/ directory (e.g. datasets/schemas/v2_event_sequence_annotation.schema.json, datasets/v2_external_test_policy.yaml, datasets/v2_review1_reviewed_bootstrap/)",
    "git_status_evidence": "Pre-existing git status confirmed uncommitted deletions of datasets/v2_review1_reviewed_bootstrap/ images and labels",
    "compliance_rule": "Per prompt Phase 2 and Phase 11, do not run git reset --hard, do not restore user data against policy, and classify failures clearly rather than faking passes"
  },
  "environment": {
    "os": "Windows",
    "python": "3.11.9 (64-bit)",
    "torch": "2.13.0+cpu",
    "cuda_available": false,
    "docker_available": false
  }
}
```

Test failures caused by the pre-existing deleted `datasets/**` tree are separated from V7 code regressions. Docker availability is reported, not assumed.
