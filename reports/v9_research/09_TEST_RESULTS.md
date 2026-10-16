# V9 test results

Overall status: **PASS**.

| Check | Status | Seconds | Evidence |
| --- | --- | --- | --- |
| compile | PASS | 0.37 | `reports/v9_research/compile.log` |
| ruff_critical | PASS | 0.27 | `reports/v9_research/ruff_critical.log` |
| ruff_v9 | PASS | 0.17 | `reports/v9_research/ruff_v9.log` |
| pytest_v9 | PASS | 11.40 | `reports/v9_research/pytest_v9.log` |
| pytest_full | PASS | 63.52 | `reports/v9_research/pytest_full.log` |
| frontend_build | PASS | 13.05 | `reports/v9_research/frontend_build.log` |

- V9 targeted pytest: 9 tests, 0 failures,
  0 errors, 0 skipped.
- Full pytest: 561 tests, 0 failures,
  0 errors, 0 skipped.

These checks validate software behavior and report consistency. They do not establish independent
model accuracy or production readiness.
