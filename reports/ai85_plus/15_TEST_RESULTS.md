# Verification results

| Check | Result | Evidence |
| --- | --- | --- |
| compile | PASS | compile_20261016T103107Z.log |
| ruff_critical | PASS | ruff_critical_20261016T103107Z.log |
| ruff_new_core | PASS | ruff_new_core_20261016T103107Z.log |
| targeted | PASS | targeted_20261016T103107Z.log |
| full_pytest | PASS | full_pytest_20261016T103107Z.log |
| frontend_build | PASS | frontend_build_20261016T103107Z.log |
| backend_smoke | PASS | backend_smoke.json |
| inference_smoke | PASS | runtime_benchmark.json; phone_spatial_probes.json; seatbelt_results.json |
| browser_E2E | BLOCKED | No configured browser E2E harness |

Full pytest: **552 PASSED, 0 FAILED, 0 ERROR, 0 SKIPPED**. Targeted checks: 32 passed. Browser E2E is one separately BLOCKED check, not a skipped or passing pytest case.

Full pytest counts are parsed from JUnit XML rather than terminal progress. Earlier failed attempts are retained; final counts refer to the final run. The prior 37 failures / 74 errors from missing dataset paths were resolved by root-aware schema/policy consumers. The readiness test previously skipped due to its stale path is now executed.
Backend smoke uses temporary SQLite/uploads/evidence and checks real startup, liveness, readiness, OpenAPI and authentication rejection. Full pytest covers queue/review/evidence and split guards. Frontend TypeScript/Vite build completed successfully. Browser-driven E2E remains BLOCKED because no browser E2E harness is configured; API/inference smoke is reported separately.
No test was deleted or weakened to improve PASS counts. Ground truth and event matcher remain unchanged. [Machine test results](test_results.json).
