# 11. Temporal Engine Regression & Fixes
- **Bug Fixed:** In `training/dmd/online_candidate.py`, removed `self.engine = TemporalEventEngine()` on unadmitted frames.
- **Impact:** Preserves temporal decay and window bridging during momentary occlusions or track loss.
- **Test Passed:** `tests/test_v18_bug_fixes.py::test_e2_temporal_state_not_reset_on_single_unadmitted_frame`.
