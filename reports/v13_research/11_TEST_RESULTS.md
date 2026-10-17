# V13 test results

Python compile and scoped Ruff passed. Targeted V11/V12/V13-adjacent tests: 44 passed, 0 failures, 0 errors. Full pytest: 619 passed, 0 failures, 0 errors. Frontend build: PASS. Backend OpenAPI smoke: PASS.

New coverage verifies original-resolution pixel preservation, deterministic frame hashing, identity scale, boundary padding and partial phone coverage. Existing tests continue to cover rights gates, protected subjects, OnlineE3 parity/reset, numeric model loading, association ambiguity and stable event IDs.
