# V11 test results

| Check | Result | Evidence |
| --- | --- | --- |
| Python compile | PASS | `py -m compileall -q backend training tests`; log SHA-256 `e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855` |
| Scoped Ruff | PASS | V11 modules and tests; log SHA-256 `af352a86840ad0af3d37ea0d9197f868363a6dac6b1c2ef5d2722f6b41d2d9af` |
| Targeted V11 pytest | PASS | 14 tests, 11.50s, 0 failures/errors |
| Full pytest | PASS | 589 tests, 70.99s, 0 failures/errors |
| Frontend production build | PASS | Vite build completed |
| Backend smoke | PASS | OpenAPI application construction returned PASS |
| Artifact verification | PASS after final generation | hashes, governance, prediction artifacts, and report contract |
| Inference smoke | NOT RERUN | no active pipeline change; inherited measured runtime retained |

Targeted tests cover past-only causal representations and normalization, stable feature dimensions, ROI coordinate transforms and border padding, missing evidence, mounted/static devices, passenger phones, verified/unverified occupant logic, dataset-rights enforcement, protected-subject rejection, and deterministic metrics. Full tests were run without deleting, skipping, or relaxing existing checks. Machine-readable JUnit, logs, and SHA-256 values are stored beside this report.

The inference result is not reported as PASS because V11 did not load the full multi-model pipeline. The runtime report clearly separates cached feature probes from measured full-backend FPS.
