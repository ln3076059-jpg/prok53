# ROADWATCH V19 — V17/V18 Report Reliability & Claim Audit
## 1. Ten Mandatory Claim Verifications
1. **13 FN Cases Tracked:** `VERIFIED_FROM_RAW_OUTPUT`. Documented in `phone_proposal_mining.json` with exact case IDs.
2. **Short Events (<1.2s):** `VERIFIED_FROM_CODE`. Dropped by default `min_positive_seconds=1.2` in `temporal_config_v2.json`.
3. **Detector Misses (11/13):** `VERIFIED_FROM_CODE`. Confirmed by downscaling to 640x640 reducing phone pixel area to <20px.
4. **4 False Positives:** `VERIFIED_FROM_RAW_OUTPUT`. Caused by mounted phone proximity in `gC-14` and event boundary fragmentation.
5. **Runtime 28.1 FPS:** `CONTRADICTED`. Refers to cached feature matrix reading; real CPU full-frame inference is ~1.0-2.2 FPS.
6. **Detector Crop vs SAHI:** `VERIFIED_FROM_CODE`. V14/V18 implemented native-resolution driver crop proposals, not multi-grid SAHI.
7. **GT-aware vs Autonomous:** `VERIFIED_FROM_CODE`. Autonomous adapter strictly rejects ground-truth and oracle metadata.
8. **Bug Fixes "Resolved" Event F1:** `CONTRADICTED`. V18 proved regression tests (5/5 pass), but real event replay was unmeasured.
9. **Single-Frame Result as Model Improvement:** `CONTRADICTED`. Inspecting frame at 60s in `gA-1` was a diagnostic check, not an accuracy measurement.
10. **Data Rights Approval:** `VERIFIED_FROM_CODE`. Live permissions correctly maintain `production_approved: false`.
