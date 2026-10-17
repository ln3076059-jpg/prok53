# V11 event metrics

| Scope | TP | FP | FN | Precision | Recall | F1 | FA/min |
| --- | --- | --- | --- | --- | --- | --- | --- |
| Frozen controlled binary, 29 events | 16 | 4 | 13 | 80.00% | 55.17% | 65.31% | 0.1813 |
| Inherited context-naive replay, two subjects | 6 | 4 | 14 | 60.00% | 30.00% | 40.00% | 0.2620 |
| Inherited context-aware replay, two subjects | 6 | 3 | 14 | 66.67% | 30.00% | 41.38% | 0.1965 |

V11 E3 is a clip classifier with Macro-F1 56.39%; it has no Event F1. Phone AP50, overall mAP50, and mAP50-95 are NOT MEASURED because the reviewed FN lane is not a complete detection test set. No 85% target is met. Event match evidence explicitly records zero V11 recovered/lost/new event IDs and preserves the frozen ground truth and matcher.
