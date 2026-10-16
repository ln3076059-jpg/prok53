# Measured metrics comparison

| Experiment | TP | FP | FN | Precision | Recall | F1 | FA/min | SAFE | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V7_A_V6_BASELINE_ACTION | 15 | 5 | 14 | 75.00% | 51.72% | 61.22% | 0.2266 | 12.57% | 34.23% |
| V7_A_V6_BASELINE_FUSION | 16 | 5 | 13 | 76.19% | 55.17% | 64.00% | 0.2266 | 12.57% | 34.23% |
| BASELINE_NEGATIVE_FUSION | 16 | 5 | 13 | 76.19% | 55.17% | 64.00% | 0.2266 | 12.57% | 34.23% |
| MEAN_FINETUNE_ACTION | 15 | 6 | 14 | 71.43% | 51.72% | 60.00% | 0.2719 | 12.13% | 34.16% |
| MEAN_FINETUNE_FUSION | 16 | 6 | 13 | 72.73% | 55.17% | 62.75% | 0.2719 | 12.13% | 34.16% |
| MEAN_FINETUNE_NEGATIVE_FUSION | 16 | 6 | 13 | 72.73% | 55.17% | 62.75% | 0.2719 | 12.13% | 34.16% |
| ATTENTION_FINETUNE_ACTION | 15 | 6 | 14 | 71.43% | 51.72% | 60.00% | 0.2719 | 12.13% | 34.11% |
| ATTENTION_FINETUNE_FUSION | 16 | 6 | 13 | 72.73% | 55.17% | 62.75% | 0.2719 | 12.13% | 34.11% |
| ATTENTION_FINETUNE_NEGATIVE_FUSION | 16 | 6 | 13 | 72.73% | 55.17% | 62.75% | 0.2719 | 12.13% | 34.11% |
| NATIVE112_FINETUNE_ACTION | 13 | 7 | 16 | 65.00% | 44.83% | 53.06% | 0.3173 | 19.01% | 40.21% |
| NATIVE112_FINETUNE_FUSION | 14 | 7 | 15 | 66.67% | 48.28% | 56.00% | 0.3173 | 19.01% | 40.21% |
| NATIVE112_FINETUNE_NEGATIVE_FUSION | 14 | 7 | 15 | 66.67% | 48.28% | 56.00% | 0.3173 | 19.01% | 40.21% |

Complete per-subject metrics, confusion matrices, event predictions and matching evidence: `combined_metrics.json`. Recovered/lost truths, FP differences and nominal 95% Wilson intervals: `event_deltas_and_intervals.json`. The intervals assume independent events and are optimistic for three subjects. Clip macro-F1 does not replace event F1.
