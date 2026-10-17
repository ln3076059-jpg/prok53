# V12 online candidate

`OnlineE3` reproduces spatial-plus-past-running normalization one sample at a time. It resets on session/camera/track changes, missing views, out-of-order timestamps, and cadence gaps. State is bounded to 8 frames and 120 feature rows. Eight initial samples per stream are marked insufficient-history; they are not silently interpreted as negative evidence.

| Subject | Samples | Parity | Max error | p50 ms | p95 ms | p99 ms |
| --- | --- | --- | --- | --- | --- | --- |
| gB-9 | 809 | True | 0.00e+00 | 3.397 | 4.419 | 4.910 |
| gC-14 | 795 | True | 9.31e-10 | 3.396 | 4.506 | 5.355 |
| gZ-36 | 1024 | True | 0.00e+00 | 3.387 | 5.094 | 6.767 |

These timings cover cached 64×64 RGB to E3 feature only. They exclude decode, detector, fitted head, fusion and event persistence. No fitted E3 scaler/head exists in V11 artifacts, so real E3 event replay remains `BLOCKED_MISSING_FITTED_HEAD_SCALER`. V12 provides JSON-only numeric export/load code and verifies it on synthetic fixtures; it does not manufacture model parameters. Probability output is explicitly uncalibrated. The research adapter remains disabled unless a caller opts in.
