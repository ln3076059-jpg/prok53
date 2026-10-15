# Phone Ablation Results

## Clip/coarse classification

| Variant | Six-class accuracy | Six-class macro-F1 | SAFE recall | Coarse macro-F1 |
|---|---:|---:|---:|---:|
| FLAT_UNWEIGHTED | 33.26% | 30.81% | 11.53% | 46.83% |
| FLAT_LOSS_ONLY | 36.60% | 34.62% | 1.95% | 39.83% |
| FLAT_SAMPLER_ONLY | 34.52% | 26.71% | 0.00% | 42.40% |
| HIERARCHICAL_LOSS_ONLY | 36.75% | 34.23% | 12.57% | 46.48% |

## Event selection gates

| Experiment | Event F1 | FA/min | SAFE recall | Worst-subject FA/min | Eligible |
|---|---:|---:|---:|---:|---:|
| V6_FLAT_UNWEIGHTED_ACTION | 51.16% | 0.1360 | 11.53% | 0.3490 | False |
| V6_FLAT_UNWEIGHTED_FUSION | 54.55% | 0.1360 | 11.53% | 0.3490 | False |
| V6_FLAT_LOSS_ONLY_ACTION | 50.00% | 0.1813 | 1.95% | 0.5889 | False |
| V6_FLAT_LOSS_ONLY_FUSION | 53.33% | 0.1813 | 1.95% | 0.5889 | False |
| V6_FLAT_SAMPLER_ONLY_ACTION | 62.75% | 0.2719 | 0.00% | 0.6979 | False |
| V6_FLAT_SAMPLER_ONLY_FUSION | 61.54% | 0.3173 | 0.00% | 0.6979 | False |
| V6_HIERARCHICAL_LOSS_ONLY_ACTION | 61.22% | 0.2266 | 12.57% | 0.4653 | False |
| V6_HIERARCHICAL_LOSS_ONLY_FUSION | 64.00% | 0.2266 | 12.57% | 0.4653 | False |

Selected result: không có V6 candidate vượt toàn bộ gate đăng ký trước. This is still DEVELOPMENT_ONLY and does not replace runtime V2 or historical V5 B2.

## False-positive categories across V6 fusion experiments

- OTHER_ACTIVITY_MISCLASSIFIED: 9
- SAFE_DRIVING_MISCLASSIFIED: 8

The case-level intervals, sampled-label fractions and overlapping ground truth are preserved in `v6_class_collapse_ablation.json`. PHONE_VISIBLE and ACTION_PREDICTED remain weaker evidence than VERIFIED_VIOLATION.
