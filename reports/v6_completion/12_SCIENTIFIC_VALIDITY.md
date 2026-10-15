# Scientific Validity

1. Outer evaluation is Leave-One-Subject-Out over gB-9, gZ-36 and gC-14; all 29 events use unchanged truth and matching rules.
2. gZ-37 and gE-28 are consumed holdouts and were not accessed by V6 development.
3. The V6 threshold (0.80) and selection gates were fixed before completed V6 results.
4. V5 `_calibrate_threshold()` did not consume outer-fold labels, but its calibration predictions came from fitted training subjects. This is training-only calibration with optimistic bias, not reliable nested validation.
5. With only two training subjects inside each outer fold, a one-subject inner validation estimate would be too unstable to present as robust nested calibration.
6. Checkpoint, input and configuration hashes are retained. Fold training ran in isolated processes after a documented RAM failure; batch size was changed before completed results, not after inspecting metrics.
7. Seatbelt results come from its own validation predictions and are never conflated with DMD phone-use metrics.
8. Cabin-video evidence is not generalized to roadside, overhead or traffic-scene cameras.
9. Every action/fusion alert remains NEEDS_REVIEW/PENDING; no AI label was converted to HUMAN_APPROVED.

Readiness: **Share with caveats** for thesis development evidence; **not ready** for production or a new unbiased performance claim.
