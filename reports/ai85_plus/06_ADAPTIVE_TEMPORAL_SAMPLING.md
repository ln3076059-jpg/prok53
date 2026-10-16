# T0–T5 controlled temporal sampling

T0: original 15-frame stride. T1: 3-frame stride. T2: baseline plus current motion triggers. T3: baseline plus 6-frame stride. T4: motion-triggered 0.4s sampling burst. T5: T4 with two-sample causal confirmation, no confirmation-stage future smoothing or endpoint extrapolation. Inherited object/action late fusion remains common to all variants.
Motion is the normalized mean absolute grayscale difference on the 3-frame observation grid. Threshold 0.035 and burst length were engineering constants frozen before the experiment; they were not searched by GT labels.
The sampler is prefix-causal and has no annotation input. The inherited centered BiGRU is not causal: each action center uses ±60 original frames, i.e. 2.016s of future context. T0–T4 additionally inherit one-sample centered smoothing (about 0.504s at T0). Matcher onset latency is not usable online alarm delay; context availability must also be counted. T5 confirmation timestamps are retained where present, while object-fusion emission latency is not measured.
Original 15-frame embeddings are reused byte-for-byte. T0 probability discrepancy is at most 5.96046448e-08, and its TP/FP/FN reproduce 16/5/13.

| Variant | <1s matched/total | 1–3s | 3–10s | >10s | Recovered | Lost | Duplicate matches |
| --- | --- | --- | --- | --- | --- | --- | --- |
| T0 | 0/4 | 0/1 | 1/1 | 15/23 | 0 | 0 | 1 |
| T1 | 0/4 | 0/1 | 1/1 | 16/23 | 1 | 0 | 5 |
| T2 | 0/4 | 0/1 | 1/1 | 15/23 | 0 | 0 | 1 |
| T3 | 0/4 | 0/1 | 1/1 | 16/23 | 1 | 0 | 3 |
| T4 | 0/4 | 0/1 | 1/1 | 15/23 | 0 | 0 | 1 |
| T5 | 0/4 | 0/1 | 1/1 | 15/23 | 0 | 0 | 2 |

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

T1/T3 recover one long event and lose zero baseline matches; subsecond recall remains 0%. T1 adds eight net FP, so its F1 is lower. Exact new/removed FP intervals, per-subject scores, masks and full match evidence are in [temporal_results.json](temporal/temporal_results.json).
All contexts were extracted in the benchmark. Selected centers alone cannot be presented as actual encoder savings or full-system FPS.
