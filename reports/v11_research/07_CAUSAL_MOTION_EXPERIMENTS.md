# V11 causal motion experiments

`CausalMotionEncoder` uses only the current feature and historical lags. The pre-registered lags 1, 3, and 7 correspond to about 0.504, 1.512, and 3.528 seconds in the frozen cache. Tests prove that changing future rows cannot change prior outputs. Causal running standardization also computes every row from prior history only.

| ID | Macro-F1 | Balanced acc. | SAFE R | PHONE R | OTHER R | Worst SAFE | Worst Macro-F1 | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V11-A_FROZEN_RESNET | 17.56% | 32.35% | 6.82% | 6.53% | 83.70% | 0.00% | 14.63% | REJECT |
| V11-B_CAUSAL_LAG1 | 47.57% | 51.21% | 37.74% | 50.10% | 65.79% | 9.96% | 39.88% | REJECT |
| V11-C_CAUSAL_MULTISCALE | 53.85% | 54.47% | 37.16% | 69.91% | 56.34% | 0.83% | 43.18% | REJECT |
| V11-D_BEHAVIOR_MULTI_ROI | 44.63% | 52.87% | 49.35% | 23.14% | 86.12% | 3.32% | 11.95% | REJECT |
| V11-E_ROI_PLUS_CAUSAL | 56.60% | 62.00% | 55.30% | 43.78% | 86.92% | 0.00% | 9.97% | REJECT |

Lag-1 motion improves pooled Macro-F1 over the frozen appearance baseline, and multi-scale causal motion improves it further to 53.85%. However, its worst-subject SAFE recall is only 0.83%. Every initial representation candidate therefore failed the worst-subject stop rule. No future frame, bidirectional temporal head, or ground-truth boundary was used. No event predictions were generated from these clip scores.
