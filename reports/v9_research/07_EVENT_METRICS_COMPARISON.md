# V9 event metrics comparison

| Experiment | TP | FP | FN | P | R | F1 | FA/min | ΔTP/ΔFP/ΔFN |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Historical V6 / T0 | 16 | 5 | 13 | 76.19% | 55.17% | 64.00% | 0.2266 | +0/+1/+0 |
| AI85 next controlled binary | 16 | 4 | 13 | 80.00% | 55.17% | 65.31% | 0.1813 | +0/+0/+0 |
| AI85 next exploratory | 16 | 2 | 13 | 88.89% | 55.17% | 68.09% | 0.0906 | +0/-2/+0 |
| T1 | 17 | 13 | 12 | 56.67% | 58.62% | 57.63% | 0.5892 | +1/+9/-1 |
| T2 | 16 | 6 | 13 | 72.73% | 55.17% | 62.75% | 0.2719 | +0/+2/+0 |
| T3 | 17 | 11 | 12 | 60.71% | 58.62% | 59.65% | 0.4986 | +1/+7/-1 |
| T4 | 16 | 7 | 13 | 69.57% | 55.17% | 61.54% | 0.3173 | +0/+3/+0 |
| T5 | 16 | 8 | 13 | 66.67% | 55.17% | 60.38% | 0.3626 | +0/+4/+0 |

No eligible development result meets P/R/F1 ≥85%, FA/min ≤0.15 and SAFE recall ≥85%. The controlled
binary comparator remains TP=16, FP=4, FN=13, P=80.00%, R=55.17%, F1=65.31%, FA/min=0.1813.
Independent validation is absent; no development value is a production claim.
