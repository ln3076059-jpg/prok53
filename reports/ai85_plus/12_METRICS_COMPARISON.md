# Comparable event metrics

| Candidate | TP | FP | FN | P | R | F1 | FA/min | Coarse SAFE R |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Historical V6 / T0 | 16 | 5 | 13 | 76.19% | 55.17% | 64.00% | 0.2266 | 12.57% |
| AI85 next controlled binary | 16 | 4 | 13 | 80.00% | 55.17% | 65.31% | 0.1813 | not comparable |
| AI85 next exploratory | 16 | 2 | 13 | 88.89% | 55.17% | 68.09% | 0.0906 | not comparable |
| T1 | 17 | 13 | 12 | 56.67% | 58.62% | 57.63% | 0.5892 | 12.36% |
| T2 | 16 | 6 | 13 | 72.73% | 55.17% | 62.75% | 0.2719 | 12.61% |
| T3 | 17 | 11 | 12 | 60.71% | 58.62% | 59.65% | 0.4986 | 12.43% |
| T4 | 16 | 7 | 13 | 69.57% | 55.17% | 61.54% | 0.3173 | 11.85% |
| T5 | 16 | 8 | 13 | 66.67% | 55.17% | 60.38% | 0.3626 | 11.85% |

Historical direct coarse SAFE recall 70.81% is a separate experiment and is not attached to the binary event row. Six-class action and seatbelt metrics remain separate from event metrics. The exploratory 88.89% precision row is ineligible for selection.
Per-subject event evidence and recovered/lost event identities are in temporal/temporal_results.json. T1/T3 recover one long event without losing baseline TP, but add FP and decrease F1. No new eligible candidate is selected. [Machine table](metrics_comparison.json).
