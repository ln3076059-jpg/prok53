# Roadwatch V4 Metrics Comparison

| Result | Dataset role | Precision | Recall | F1 | FA/min | TP | FP | FN |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| V3 historical | consumed holdout `gE-28` | 30.8% | 40.0% | 34.8% | 1.116 | 4 | 9 | 6 |
| V3.1 B5 historical | development pool | 37.5% | 10.3% | 16.2% | 0.227 | 3 | 5 | 26 |
| V4 candidate | development pool | NOT RUN | NOT RUN | NOT RUN | NOT RUN | — | — | — |
| V4 candidate | new independent holdout | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | NOT AVAILABLE | — | — | — |

V3 and B5 are shown for provenance, not as a before/after comparison: they use different datasets.
No V4 accuracy delta can be claimed until the development caches are restored from an authorized
source and evaluated under the frozen protocol.

`NEW_INDEPENDENT_HOLDOUT = NOT_AVAILABLE`

`V4_GENERALIZATION_CLAIM = NOT_ESTABLISHED`
