# Runtime and tests

Machine-readable command results are in `test_results.json`; raw output is retained in the
adjacent `.log` files. `reproduce.ps1` reruns the audit, report renderer, compilation, Ruff,
targeted tests, full tests, backend smoke, frontend build and bundle verification.

No model inference or fitting ran in V16, so a new detector p50/p95, RAM, VRAM or training
throughput number does not exist. Historical runtime is not relabeled as a V16 benchmark.
The V16 engineering change is exercised without any network access by the fail-closed downloader
test and by annotation/split validation tests.
