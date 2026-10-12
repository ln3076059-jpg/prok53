# Roadwatch V4 Test Results

Executed on 2026-10-12 with Python 3.11.9 and Node 22.17.1.

| Check | Result |
|---|---|
| CI Ruff gate (`E9,F63,F7,F82`) | PASS |
| Python compileall | PASS |
| FastAPI import smoke | PASS |
| Frontend `npm run build` | PASS |
| New engineering + V4 tests | 33 passed |
| Dataset-independent regression suite | 287 passed, 1 skipped |
| Full pytest | 389 passed, 1 skipped, 37 failed, 74 errors; blocked by missing tracked `datasets/` schemas and policies |

Regression coverage added for RBAC, authenticated-export support code, video signatures, stale
concurrent review rejection, durable queue claiming, environment gating, six-state phone context,
strict abstention, stable occupant association across track changes, temporal state/deduplication,
V4 config production blocking, and development-only experiment preflight.

The full suite was not skipped: it was executed and produced failures/errors dominated by missing
`datasets/schemas/v2_event_sequence_annotation.schema.json`, `datasets/schemas/v2_review1.schema.json`,
`datasets/v2_external_test_policy.yaml` and related pre-existing deletions.
