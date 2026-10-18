# ROADWATCH V20 — Runtime & Test Results
## 1. Test Verification
- V18 Regression Tests: 5/5 PASS.
- V19 Regression & Reconciliation Tests: 6/6 PASS.
- V20 Scientific Engineering & Governance Tests: 8/8 PASS.
- Total Targeted Governance Tests: 19/19 PASS (100%).
- Full Test Suite Baseline: 659 tests PASS.
- Python syntax compilation (`compileall`): ALL PASS.
- Scoped Ruff lint: ALL PASS (0 errors).

## 2. Hardware Runtime Benchmarks (CPU, Windows)
- Full-frame YOLO specialist: ~610 ms/frame (~1.6 FPS).
- Driver-crop YOLO specialist: ~380 ms/frame (~2.6 FPS).
- Pose estimator (when candidate detected): ~3040 ms/frame.
- End-to-end autonomous video inference: 3.19 - 3.76 effective FPS on CPU at stride 15 (2 Hz sampling).
