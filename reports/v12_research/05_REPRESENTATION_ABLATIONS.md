# V12 representation ablations

No new representation was fit because the authoritative DMD record denies fitting. V12-A retained the frozen V11 predictions only after exact metric reconciliation. V12-B retained the online implementation because it matches the batch E3 features across all three caches. V12-C retained original-resolution ROI as a diagnostic hypothesis because reviewed boxes become extremely small at 64×64.

| Held-out subject | Macro-F1 | SAFE recall | PHONE recall | OTHER recall |
| --- | --- | --- | --- | --- |
| gB-9 | 62.24% | 65.42% | 72.84% | 46.05% |
| gC-14 | 55.47% | 61.54% | 61.78% | 44.71% |
| gZ-36 | 52.14% | 54.36% | 66.06% | 39.43% |

V12-D appearance/motion fitting, V12-G temporal fitting, calibration and learned fusion are BLOCKED. The reported folds are repeatedly examined LEVEL 2 development evidence and were already used for E3 selection. They are not a new unbiased test. No threshold sweep, seed sweep, Group DRO, CLIP download, backbone tuning, or protected-holdout access was performed.
