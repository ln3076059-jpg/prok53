# V12 runtime and tests

| Check | Result |
| --- | --- |
| Python compile | PASS; exit code 0 |
| Scoped Ruff | PASS; exit code 0 |
| Targeted V11+V12 pytest | PASS: 43 tests, 0 failures, 0 errors |
| Full pytest | PASS: 618 tests, 0 failures, 0 errors |
| Frontend build | PASS |
| Backend OpenAPI smoke | PASS |
| Full V12 inference smoke | NOT RUN: no fitted E3 head/scaler and no active-pipeline change |

Streaming E3 feature p50 across subjects is 3.397 ms; maximum subject p95 is 5.094 ms and p99 is 6.767 ms. This is cached feature latency only. The inherited full-backend benchmark remains 1.119 FPS with p50 600.79 ms and p95 2050.07 ms. V12 does not claim real-time operation.
