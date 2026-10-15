# Metrics Comparison

## Historical verified baselines

| Experiment | TP | FP | FN | Precision | Recall | F1 | FA/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| B0_V4_CURRENT | 3 | 5 | 26 | 37.50% | 10.30% | 16.20% | 0.2270 |
| B2_TINY_CNN_TEMPORAL | 6 | 5 | 23 | 54.55% | 20.69% | 30.00% | 0.2266 |
| B3_RESNET18_GRU | 10 | 7 | 19 | 58.82% | 34.48% | 43.48% | 0.3173 |
| B8_B3_LATE_FUSION | 11 | 7 | 18 | 61.11% | 37.93% | 46.81% | 0.3173 |

## V6 fixed-threshold experiments

| Experiment | TP | FP | FN | Precision | Recall | F1 | FA/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| V6_FLAT_UNWEIGHTED_ACTION | 11 | 3 | 18 | 78.57% | 37.93% | 51.16% | 0.1360 |
| V6_FLAT_UNWEIGHTED_FUSION | 12 | 3 | 17 | 80.00% | 41.38% | 54.55% | 0.1360 |
| V6_FLAT_LOSS_ONLY_ACTION | 11 | 4 | 18 | 73.33% | 37.93% | 50.00% | 0.1813 |
| V6_FLAT_LOSS_ONLY_FUSION | 12 | 4 | 17 | 75.00% | 41.38% | 53.33% | 0.1813 |
| V6_FLAT_SAMPLER_ONLY_ACTION | 16 | 6 | 13 | 72.73% | 55.17% | 62.75% | 0.2719 |
| V6_FLAT_SAMPLER_ONLY_FUSION | 16 | 7 | 13 | 69.57% | 55.17% | 61.54% | 0.3173 |
| V6_HIERARCHICAL_LOSS_ONLY_ACTION | 15 | 5 | 14 | 75.00% | 51.72% | 61.22% | 0.2266 |
| V6_HIERARCHICAL_LOSS_ONLY_FUSION | 16 | 5 | 13 | 76.19% | 55.17% | 64.00% | 0.2266 |

Research targets are P≥60%, R≥40%, F1≥50%, FA/min≤0.30. They are targets, not claims. The registered gate result is không có V6 candidate vượt toàn bộ gate đăng ký trước.

All event comparisons use the same 29-event truth and matching policy. V5 calibration used predictions from subjects fitted by each fold, so it did not leak outer heldout labels but was optimistically in-sample. V6 does not tune its threshold per fold or variant.
