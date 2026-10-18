# ROADWATCH V20 — Temporal Event Replay & Reclassification
## 1. Classification of Replay Results
| Run Identifier | Evaluation Type | TP | FP | FN | Precision | Recall | F1 | FA/min | Status |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **R0** | SIMULATED_BUG_REPLAY | 4 | 15 | 16 | 21.05% | 20.00% | 20.51% | 0.9823 | Simulated pre-V18 engine reset fragmentation |
| **R3** | FROZEN_OBSERVATION_REPLAY | 6 | 3 | 14 | 66.67% | 30.00% | 41.38% | 0.1965 | Replay on frozen V9 cached observations |
| **R4** | GT_INFORMED_DIAGNOSTIC | 7 | 4 | 13 | 63.64% | 35.00% | 45.16% | 0.2620 | Reclassified: uses FN-mined proposals |
| **R5** | GT_INFORMED_DIAGNOSTIC | 7 | 4 | 13 | 63.64% | 35.00% | 45.16% | 0.2620 | Reclassified: identical to R4 |
| **V20** | **AUTONOMOUS_VIDEO_INFERENCE** | **6** | **16** | **14** | **27.27%** | **30.00%** | **28.57%** | **1.0478** | **True autonomous video inference (15.27 min)** |

## 2. Autonomous Evaluation Breakdown
- **Cohort:** `gC-14` + `gZ-36` (20 Ground Truth distraction events, 15.27 evaluated minutes).
- **gC-14:** TP=2, FP=8, FN=7, Precision=20.00%, Recall=22.22%, F1=21.05%, Effective FPS: 3.19.
- **gZ-36:** TP=4, FP=8, FN=7, Precision=33.33%, Recall=36.36%, F1=34.78%, Effective FPS: 3.76.
- **Key Insight:** True autonomous video inference achieves 30.00% recall (matching R3's true 6 TP without oracle input). However, false alarms are higher (16 FP, 1.048 FA/min) because inference runs over all 15.27 minutes of pixels without an action classifier (`NumericHead`) gating face/hand gestures. In contrast, R4/R5 artificially suppressed false alarms by injecting proposals only at GT timestamps.
