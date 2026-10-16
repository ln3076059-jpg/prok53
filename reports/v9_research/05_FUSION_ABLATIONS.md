# V9 fusion ablations

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

T1/T3 recover one event but add 9/7 false positives versus the controlled comparator. T2–T5 do not
improve F1. The post-hoc row remains ineligible. Missing camera views are excluded from the denominator;
secondary cameras, pose and negative evidence cannot create an alert without physical and driver context.


## Strict phone-context diagnostic

At the 0.02 proposal-mining floor, strict context gating removes
99/226 mounted/static proposals
(43.81%) and retains
12/12 human-selected handheld
boxes. It rejects all proposals in the visible-phone hard example `gB-9-phone-06`. This is candidate
yield on FN-selected positive development evidence, not detector accuracy or event P/R/F1.
