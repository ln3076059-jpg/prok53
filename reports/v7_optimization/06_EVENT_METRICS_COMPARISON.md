# Event metrics comparison

| Fusion experiment | TP | FP | FN | Precision | Recall | F1 | FA/min | SAFE recall | Worst subject FA/min | Gates |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---|
| V7_A_V6_BASELINE_FUSION | 16 | 5 | 13 | 76.19% | 55.17% | 64.00% | 0.2266 | 12.57% | 0.4653 | FAIL |
| V7_B_COARSE_OBJECTIVE_FUSION | 18 | 11 | 11 | 62.07% | 62.07% | 62.07% | 0.4986 | 0.15% | 0.8142 | FAIL |
| V7_C_HARD_NEGATIVES_FUSION | 17 | 11 | 12 | 60.71% | 58.62% | 59.65% | 0.4986 | 2.54% | 1.1632 | FAIL |
| V7_D_BINARY_GATE_FUSION | 15 | 8 | 14 | 65.22% | 51.72% | 57.69% | 0.3626 | 1.05% | 0.5816 | FAIL |
| V7_E_LONGER_CONTEXT_FUSION | 14 | 10 | 15 | 58.33% | 48.28% | 52.83% | 0.4532 | 0.00% | 0.5889 | FAIL |
| V7_F_NEGATIVE_MARGIN_FUSION | 19 | 15 | 10 | 55.88% | 65.52% | 60.32% | 0.6799 | 0.15% | 1.0489 | FAIL |

Selection uses the unchanged V6 gates: FA/min ≤0.30, worst-subject FA/min ≤0.60, SAFE recall ≥20%, then maximum event F1. Selected candidate: `NONE`.
