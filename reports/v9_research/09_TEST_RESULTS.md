# V9 test results

Overall status: **PASS**.

| Check | Status | Seconds | Evidence |
| --- | --- | --- | --- |
| compile | PASS | 0.52 | `reports/v9_research/compile.log` |
| ruff_critical | PASS | 0.29 | `reports/v9_research/ruff_critical.log` |
| ruff_v9 | PASS | 0.20 | `reports/v9_research/ruff_v9.log` |
| pytest_v9 | PASS | 9.91 | `reports/v9_research/pytest_v9.log` |
| pytest_full | PASS | 66.74 | `reports/v9_research/pytest_full.log` |

- V9 targeted pytest: 7 tests, 0 failures,
  0 errors, 0 skipped.
- Full pytest: 559 tests, 0 failures,
  0 errors, 0 skipped.

These checks validate software behavior and report consistency. They do not establish independent
model accuracy or production readiness.
