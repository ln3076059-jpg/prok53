# V10 test results

| Check | Result | Evidence |
| --- | --- | --- |
| Python compile | PASS | `python -m compileall -q backend training tests` |
| Scoped Ruff | PASS | V10 scripts and tests |
| Targeted pytest | PASS | 90 tests, 39.02s, 0 failures/errors |
| Full pytest | PASS | 575 tests, 71.82s, 0 failures/errors |
| Frontend production build | PASS | TypeScript + Vite, 952 modules |
| Backend smoke | PASS | OpenAPI generated with 17 paths |
| V10 artifact verification | PASS | hashes/governance verified after report generation |
| New inference smoke | NOT RERUN | insufficient free RAM after tests; inherited measured runtime retained |

The targeted set covers V10 artifacts, protected-holdout guards, event-evaluation governance, exact occupant
context, missing views, phone geometry/context, short-event sampling, and seatbelt uncertainty. No test was
deleted or relaxed. Full results are stored in `pytest_full.xml`; targeted results are in
`pytest_targeted.xml`. The frontend build and backend smoke logs are preserved beside this report.
