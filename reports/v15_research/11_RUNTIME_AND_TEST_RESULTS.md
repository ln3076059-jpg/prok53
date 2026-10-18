# Runtime and test results

| Check | Result |
|---|---|
| Targeted pytest | PASS: 40 tests, 0 failures, 0 errors |
| Full pytest | PASS: 629 tests, 0 failures, 0 errors |
| Python compile | see `compile.log` |
| Ruff | see `ruff.log` |
| Frontend build | see `frontend_build.log` |
| Backend smoke | see `backend_smoke.log` |

The only new timing is synthetic adapter geometry: p50/p95/p99 0.186/0.308/0.464 ms for 6 tiles, excluding model inference. It is not detector FPS, offline throughput, live latency, or full-pipeline FPS. Full decode→evidence CPU/GPU/RAM/VRAM profiling was NOT RUN because no valid new candidate model exists.
