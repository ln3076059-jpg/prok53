# V11 subject generalization

| ID | Macro-F1 | Balanced acc. | SAFE R | PHONE R | OTHER R | Worst SAFE | Worst Macro-F1 | Decision |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| E0_RAW_CAUSAL_MULTISCALE | 53.85% | 54.47% | 37.16% | 69.91% | 56.34% | 0.83% | 43.18% | REJECT |
| E1_SPATIAL_RESIDUAL_NORM | 51.59% | 54.30% | 40.78% | 56.91% | 65.19% | 2.90% | 42.02% | REJECT |
| E2_CAUSAL_RUNNING_NORM | 55.24% | 55.33% | 54.14% | 65.18% | 46.68% | 48.33% | 53.56% | REJECT |
| E3_SPATIAL_PLUS_RUNNING | 56.39% | 56.81% | 60.38% | 66.78% | 43.26% | 54.36% | 52.14% | SELECTED |
| E4_SUBJECT_CLASS_BALANCED | 54.67% | 56.25% | 38.90% | 67.27% | 62.58% | 2.07% | 44.73% | REJECT |

Selected E3 per-subject results:

| Held-out subject | Macro-F1 | SAFE recall | PHONE recall | OTHER recall |
| --- | --- | --- | --- | --- |
| gB-9 | 62.24% | 65.42% | 72.84% | 46.05% |
| gC-14 | 55.47% | 61.54% | 61.78% | 44.71% |
| gZ-36 | 52.14% | 54.36% | 66.06% | 39.43% |

E3 combines spatial residual normalization with past-only running normalization. Relative to raw causal E0, pooled Macro-F1 changes from 53.85% to 56.39%, worst-subject SAFE recall from 0.83% to 54.36%, and subject silhouette from 0.1092 to -0.0047. Runtime subject identity is not an input.

E3 passes the pre-registered development stability gate and is kept as a hypothesis. It is not promoted: three subjects are too few, all have been repeatedly examined, no confidence interval would restore independence, and no locked holdout exists. E4 balanced subject-class sampling regresses worst-subject SAFE recall, showing that balancing alone does not solve cabin/subject shift.
