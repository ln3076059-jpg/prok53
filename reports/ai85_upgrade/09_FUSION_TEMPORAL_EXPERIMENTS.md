# Fusion and temporal decisions

Compared inherited union fusion with a preregistered negative-evidence rule: veto an object event only when at least half of overlapping action samples favor SAFE over PHONE and SAFE probability is at least 0.5. Missing action evidence cannot veto. Action-only evidence remains review-only.

All runs use threshold 0.80, release ratio 0.65, two consecutive activation samples and the inherited two-second merge gap. Neither matcher nor ground-truth event boundaries changed. The veto retains original event bounds; no event stretching was introduced. Per-event veto reasons are stored.

| Experiment | Recovered GT | Lost GT | New FP | Net FP change |
| --- | --- | --- | --- | --- |
| V7_A_V6_BASELINE_ACTION | 0 | 1 | 0 | 0 |
| V7_A_V6_BASELINE_FUSION | 0 | 0 | 0 | 0 |
| BASELINE_NEGATIVE_FUSION | 0 | 0 | 0 | 0 |
| MEAN_FINETUNE_ACTION | 0 | 1 | 1 | 1 |
| MEAN_FINETUNE_FUSION | 0 | 0 | 1 | 1 |
| MEAN_FINETUNE_NEGATIVE_FUSION | 0 | 0 | 1 | 1 |
| ATTENTION_FINETUNE_ACTION | 0 | 1 | 1 | 1 |
| ATTENTION_FINETUNE_FUSION | 0 | 0 | 1 | 1 |
| ATTENTION_FINETUNE_NEGATIVE_FUSION | 0 | 0 | 1 | 1 |
| NATIVE112_FINETUNE_ACTION | 1 | 4 | 7 | 2 |
| NATIVE112_FINETUNE_FUSION | 1 | 3 | 7 | 2 |
| NATIVE112_FINETUNE_NEGATIVE_FUSION | 1 | 3 | 7 | 2 |

New FP uses diagnostic one-to-one pairing within subject and FP kind at temporal IoU >=0.5; it does not alter evaluation matching. The low-resolution continuations add a duplicate FP and do not recover a new truth. A separate temporal-policy search was not justified on three repeatedly used subjects. Onset latency in metrics is the inherited subject-mean statistic. Centered BiGRU clips also need roughly 2.016 seconds of lookahead; these are offline results, not live latency.
