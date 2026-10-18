# ROADWATCH V19 — Controlled Event Replay (R0–R5) Results
## 1. Empirical Results on Development Cohort (`gC-14` + `gZ-36`, 20 GT Events, 15.27 min)
| Run ID | Description | TP | FP | FN | Precision | Recall | Event F1 | FA/min | Key Architectural Impact |
| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | :--- |
| **R0** | Pre-V18 Frozen Baseline (E1 bug + E2 bug active) | 4 | 15 | 16 | 21.05% | 20.00% | 20.51% | 0.9823 | Massive event fragmentation (15 FP) caused by E2 engine reset |
| **R1** | Only E1 Fix Enabled (Context Normalized) | 4 | 15 | 16 | 21.05% | 20.00% | 20.51% | 0.9823 | E1 alone cannot overcome engine reset fragmentation |
| **R2** | Only E2 Fix Enabled (Engine State Preserved) | 6 | 3 | 14 | 66.67% | 30.00% | 41.38% | 0.1965 | **Major gain:** FP drops from 15 to 3 (-12 FP), F1 doubles to 41.38% |
| **R3** | E1 + E2 Fixes Combined | 6 | 3 | 14 | 66.67% | 30.00% | 41.38% | 0.1965 | Reconciles with V9 context-aware baseline |
| **R4** | E1 + E2 + Original-Res Proposals | 7 | 4 | 13 | 63.64% | 35.00% | 45.16% | 0.2620 | **Recovers +1 TP in gZ-36**, Recall increases to 35.00%, F1 = 45.16% |
| **R5** | Selected Engineering Candidate | 7 | 4 | 13 | 63.64% | 35.00% | 45.16% | 0.2620 | Stable candidate with calibrated hysteresis |

## 2. Scientific Insights
- **The E2 fix is the dominant quality driver:** Preserving temporal engine state across unadmitted frames eliminated 12 false positives caused by artificial alert fragmentation, driving Precision from 21.05% to 66.67%.
- **Original-resolution proposals recover real missed events:** Injecting driver-crop proposals unlocked an additional true positive (TP=7) without retraining weights.
