# Full cohort controlled experiments

| Run | Status | Minutes | GT | TP/FP/FN | Precision | Recall | F1 | FA/min |
|---|---|---:|---:|---|---:|---:|---:|---:|
| B0 | COMPLETE | 15.270497 | 20 | 6/16/14 | 0.272727 | 0.300000 | 0.285714 | 1.047772 |
| B1 | COMPLETE | 15.270497 | 20 | 0/0/20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B2 | COMPLETE | 15.270497 | 20 | 0/0/20 | 0.000000 | 0.000000 | 0.000000 | 0.000000 |
| B3 | BLOCKED | — | — | — | — | — | — | — |
| B4 | BLOCKED | — | — | — | — | — | — | — |
| T1 separate legacy-role boundary replay | COMPLETE | 15.270497 | 20 | 13/7/7 | 0.650000 | 0.650000 | 0.650000 | 0.458400 |

B0 is exact V20 D5 inference with full persistence and Stage2 separated. Replay parity checked on activation intervals. B1/B2 use the same actual detections/pose exposure and conservative UNKNOWN roles because no verified calibration was found. Association coverage is zero; losing all true events is a failure, not an improvement. B1 also changes proximity geometry; no isolated association-effect claim is made.

B3/B4 did not evaluate any frames or GT events. Empty prediction files have BLOCKED status sidecars and contain no negative evidence. No smoke result is used as a full result. Protocol, model/config/matcher/prediction hashes are persisted. Resume reuses completed sessions only; interrupted attempts remain intact.
