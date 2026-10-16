# V10 temporal event experiments

| Variant | TP | FP | FN | Event F1 | FA/min |
| --- | --- | --- | --- | --- | --- |
| AI85 next controlled binary | 16 | 4 | 13 | 65.31% | 0.1813 |
| T1 | 17 | 13 | 12 | 57.63% | 0.5892 |
| T2 | 16 | 6 | 13 | 62.75% | 0.2719 |
| T3 | 17 | 11 | 12 | 59.65% | 0.4986 |
| T4 | 16 | 7 | 13 | 61.54% | 0.3173 |
| T5 | 16 | 8 | 13 | 60.38% | 0.3626 |

T1 recovers one TP but adds nine FP relative to the controlled binary comparator; none of T1-T5 improves its
65.31% Event F1. Prior dense sampling still misses all four subsecond events. V10 therefore does not repeat a
fixed-density threshold sweep. D2 supplies a stronger clip-level motion hypothesis, but it is non-causal and
has no event predictions yet. Online onset/offset latency and fragmentation are **NOT MEASURED**. Ground truth,
event IoU, tolerance, and prediction-to-event semantics remain frozen.
