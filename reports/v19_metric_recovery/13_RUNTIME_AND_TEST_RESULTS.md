# ROADWATCH V19 — Runtime Performance & Test Results
## 1. Test Suite Execution
- **V18 Bug Fix Regression Tests:** 5 / 5 passed (100%).
- **V19 Regression & Governance Tests:** 6 / 6 passed (100%).
- **Full Pytest Suite:** 659 passed.
- **Ruff Lint Check:** Passed cleanly (0 errors).
- **Python Compilation:** Passed (`compileall -q backend training tests`).

## 2. Real Hardware Runtime (Intel Core CPU, Windows)
- Decoded Full Frame + YOLO Specialist Inference: ~480-1000 ms/frame (~1.0-2.2 FPS on CPU).
- Pre-extracted observation temporal replay: <0.2 seconds per video stream (15.27 minutes processed in 0.35s).
