# V2 Validation Calibration Protocol

Status: **UNCALIBRATED / NOT_RUN**.

Calibration uses development validation only. It must not consume frozen external-test rows.
Separate thresholds are selected for phone detection, upper-body detection, three-state belt
classification, fusion and temporal activation.

The machine-readable threshold artifact records model SHA-256, validation manifest SHA-256, score
artifact SHA-256, UTC creation time, method, objective, per-class validation counts/metrics and
`test_rows_used: 0`.

```powershell
py -m training.calibrate_thresholds reports/v2_validation_scores.csv `
  --model models/candidates/v2/phone_detector.pt `
  --validation-manifest datasets/manifests/v2_validation.jsonl `
  --output reports/v2_threshold_calibration.json
```

Phone calibration prioritizes lower false driver-phone event rate subject to declared recall.
Seatbelt calibration treats `UNCERTAIN_OR_OCCLUDED -> UNFASTENED` as safety-critical and must not
turn absent visual evidence into a violation. Threshold choices and objectives are fixed before
model lock.

Fusion training also requires a development-manifest hash. Its artifact records the fixed feature
schema/hash, normalization mean/scale, weights, intercept, threshold, development feature CSV/hash,
validation metrics and `test_rows_used: 0`. Frozen external features are prohibited.

Per-camera calibration is a separate human decision. Its record contains `camera_id`, viewpoint,
vehicle direction, driver image side, normalized windshield ROI, expected vehicle scale,
conditions, calibration date, reviewer and evidence hashes. `approved: true` is prohibited until
that record is reviewed by a named human.
