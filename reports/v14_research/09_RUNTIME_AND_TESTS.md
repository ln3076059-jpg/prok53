# V14 runtime and tests

Targeted tests: 48 passed, 0 failures, 0 errors. Full pytest: 623 passed, 0 failures, 0 errors. Python compile and scoped Ruff passed. Frontend build: PASS. Backend smoke: PASS.

The new autonomous adapter was exercised with deterministic synthetic detector fixtures, including empty proposals, forbidden oracle metadata, boundary boxes, passengers and coordinate round trips. New YOLO timing was NOT RUN due constrained host memory. Historical detector timing and V13 ROI timing retain their original scopes; neither is called end-to-end FPS.
