# End-to-End and Software Test Results

Environment: `Windows-10-10.0.26200-SP0`, Python `3.11.9`.

| Check | Status | Duration |
|---|---:|---:|
| python_compile | PASS | 0.47s |
| ruff_ci_gate | PASS | 0.39s |
| ruff_full_audit | FAIL | 0.35s |
| pytest_targeted_v5_v6 | PASS | 16.61s |
| pytest_full | FAIL | 63.32s |
| backend_import_smoke | PASS | 10.58s |
| frontend_typescript_build | PASS | 35.54s |
| docker_compose_config | NOT_AVAILABLE | 0.00s |
| pytest_backend_pipeline_expanded | PASS (112 passed, 1 skipped) | 19.68s |

The backend performance run exercised an actual DMD cabin clip through `VideoAnalyzer`, SQLite persistence and the evidence path. Research V6 models were not activated in production. The frontend build is the executable verification of TypeScript/UI integration; static UI detector findings were empty.

The full Ruff audit found 1165 issues, dominated by E501 (812), F401 (74), I001 (38) and pyupgrade-family (137). The critical CI gate count is 0; the gate command passed.

## Non-passing details

### ruff_full_audit: FAIL

```text
ue not in mapping.values():
33 |             raise ValueError("physical_vehicle_group_id must be included in vehicle_physical_groups")
   |                                                                                                     ^
34 |         parts = list(mapping.values())
35 |     result = set()
   |

E501 Line too long (102 > 100)
  --> training\validate_sequence_intake.py:44:101
   |
42 |         if dimension in {"physical_vehicle_group_id", "person_group_ids", "capture_session_id"}:
43 |             if identifier.lower().startswith("video:"):
44 |                 raise ValueError(f"{dimension} must identify a physical group, not a video namespace")
   |                                                                                                     ^^
45 |         if dimension == "sha256":
46 |             if not re.fullmatch(r"[0-9a-fA-F]{64}", identifier):
   |

E501 Line too long (109 > 100)
  --> training\validate_sequence_intake.py:69:101
   |
67 |                     development_manifest_path, development_lineage_lock_path,
68 |                 )
69 |                 if stable_json_hash(development) != stable_json_hash(read_intake(development_manifest_path)):
   |                                                                                                     ^^^^^^^^^
70 |                     raise ValueError("development rows do not match the locked input file")
71 |             except (OSError, ValueError) as exc:
   |

E501 Line too long (103 > 100)
  --> training\validate_v2_capture.py:54:101
   |
52 |         present_forbidden = forbidden & set(item)
53 |         if present_forbidden:
54 |             errors.append(f"{sample_id}: duplicated/derived sample fields {sorted(present_forbidden)}")
   |                                                                                                     ^^^
55 |         if sample_id in sample_ids:
56 |             errors.append(f"duplicate sample_id: {sample_id}")
   |

E501 Line too long (106 > 100)
   --> training\validate_v2_capture.py:112:101
    |
110 |         "errors": errors,
111 |         "warnings": warnings,
112 |         "policy": "PASS confirms declared diversity/isolation and file integrity, not label correctness.",
    |                                                                                                     ^^^^^^
113 |     }
    |

Found 1165 errors.
[*] 236 fixable with the `--fix` option (6 hidden fixes can be enabled with the `--unsafe-fixes` option).
```
### pytest_full: FAIL

```text
sequence_intake.py::test_official_freeze_requires_exact_development_lineage[wrong_lock]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[event-label]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[event-interval]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[event-remove]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[event-add]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[event-foreign]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[event-reorder]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[context-phone]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[context-seatbelt]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[context-remove]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[context-add]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[context-foreign]
ERROR tests/test_sequence_intake.py::test_final_truth_must_match_canonical_rows[context-reorder]
ERROR tests/test_sequence_intake.py::test_truth_freeze_rechecks_sources_and_evidence[event-source_changed]
ERROR tests/test_sequence_intake.py::test_truth_freeze_rechecks_sources_and_evidence[event-source_missing]
ERROR tests/test_sequence_intake.py::test_truth_freeze_rechecks_sources_and_evidence[event-review_evidence_changed]
ERROR tests/test_sequence_intake.py::test_truth_freeze_rechecks_sources_and_evidence[context-source_changed]
ERROR tests/test_sequence_intake.py::test_truth_freeze_rechecks_sources_and_evidence[context-source_missing]
ERROR tests/test_sequence_intake.py::test_truth_freeze_rechecks_sources_and_evidence[context-review_evidence_changed]
ERROR tests/test_sequence_intake.py::test_evaluator_and_integrity_verify_exact_source_set[event_set]
ERROR tests/test_sequence_intake.py::test_evaluator_and_integrity_verify_exact_source_set[context_bindings]
ERROR tests/test_sequence_intake.py::test_evaluator_and_integrity_verify_exact_source_set[source_file]
ERROR tests/test_sequence_intake.py::test_official_sequence_review_evidence_is_required[missing]
ERROR tests/test_sequence_intake.py::test_official_sequence_review_evidence_is_required[short_hash]
ERROR tests/test_sequence_intake.py::test_official_sequence_review_evidence_is_required[mismatch]
```
### docker_compose_config: NOT_AVAILABLE

```text
Docker CLI is not installed or not on PATH.
```
