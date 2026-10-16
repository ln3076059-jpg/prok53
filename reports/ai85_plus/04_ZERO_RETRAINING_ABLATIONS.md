# Measured zero-retraining probes

All weights remained frozen. Ultralytics 8.3.187; prediction confidence 0.10, class-aware NMS IoU 0.70 unless the explicit agnostic variant is used. No AP curve is calculated from this confidence cutoff.
The checkpoint training metadata says imgsz=960. The measured `A2_full_960` row is its original-resolution reference; `A0_full_640` is the matched 640 reference for the TTA/NMS probes. These names are retained in raw artifacts to avoid changing measured history. TTA was compared only at 640, not asserted at 960.
Thirteen original-video midpoints were selected for FN diagnosis, not for detector validation or model selection. All timestamps are checked inside the corresponding interval.

| Variant | Frames | Frames with proposals | Proposals | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- |
| A0_full_640 | 13 | 1 | 1 | 237.49 | 1103.71 |
| A1_full_640_TTA | 13 | 1 | 1 | 612.18 | 2024.57 |
| A2_full_960 | 13 | 0 | 0 | 469.46 | 672.68 |
| A2_full_1024 | 13 | 0 | 0 | 500.72 | 726.28 |
| A3_full_640_agnostic | 13 | 1 | 1 | 241.53 | 423.1 |
| A4_driver_640 | 13 | 2 | 2 | 289.88 | 683.18 |
| A4_padded_driver_640 | 13 | 1 | 1 | 255.14 | 587.32 |

No A5 combination was selected because no legitimate validation gain was established. No gain in phone-event recall can be inferred from proposal counts. [Raw boxes and timings](phone_spatial_probes.json).
