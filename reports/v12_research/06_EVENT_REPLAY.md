# V12 event replay

| Variant | TP | FP | FN | Precision | Recall | F1 | FA/min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen context naive, two subjects | 6 | 4 | 14 | 60.00% | 30.00% | 40.00% | 0.2620 |
| Frozen context aware, two subjects | 6 | 3 | 14 | 66.67% | 30.00% | 41.38% | 0.1965 |

The context-only delta is recovered TP IDs=[], lost TP IDs=[], removed FP IDs=['gC-14:PRED:0.000000:0.504032'], new FP IDs=[]. The removed false positive is now traceable as `gC-14:PRED:0.000000:0.504032`. Counts were recomputed from match records and duplicate predictions are charged as FP. Duration-bin records are included per subject.

The 29-event controlled comparator remains TP=16, FP=4, FN=13, P=80.00%, R=55.17%, F1=65.31%, FA/min=0.1813. Its historical scope differs from this two-subject replay. V12 generated no E3 events because the fitted head/scaler and verified occupants are absent. Emission latency was not preserved in the frozen evaluator, so online latency is NOT MEASURED rather than inferred from backdated start times.
