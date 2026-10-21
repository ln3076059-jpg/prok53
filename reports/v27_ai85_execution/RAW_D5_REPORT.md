# V27 raw D5 full-cohort report

Execution date: 2026-10-21. Scope: local development-only inference on the complete gC-14 and gZ-36 BODY videos. The run used D5 specialist-plus-auxiliary detection, verified association, stride 15, CPU, `action_mode=NONE`, and disabled action gating. No fitting or threshold search was performed.

## Metrics

| Subject | TP | FP | FN | Precision | Recall | F1 | Sampled frames | Wall time |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gC-14 | 7 | 1 | 2 | 87.50% | 77.78% | 82.35% | 795 | 334.46 s |
| gZ-36 | 7 | 2 | 4 | 77.78% | 63.64% | 70.00% | 1,024 | 425.63 s |
| **Micro** | **14** | **3** | **6** | **82.35%** | **70.00%** | **75.68%** | **1,819** | **760.09 s** |

The run covered 27,267 source frames and 15.2705 minutes of video. Sampled inference throughput was 2.39 samples/s, equivalent to 1.21 times source-video realtime at stride 15. The strict development gate remains `NOT_MET`: on 20 GT events, a feasible passing result requires at least 18 TP and at most 3 FP. The current result needs four additional TP and cannot add an FP.

## False-negative waterfall

There are six false negatives; four last less than one second. Raw evidence assigns the primary failure modes as follows:

| Primary failure | Events |
|---|---:|
| Association UNKNOWN rejected most or all evidence | 3 |
| Sparse proposals and all selected evidence rejected | 1 |
| No phone proposal in the GT interval | 1 |
| One admitted observation, insufficient temporal support | 1 |

The 12.63-second gC-14 miss is the clearest association failure: 22 proposals were recorded, 20 observations selected a candidate, but 19 were rejected as unverified occupant association. The 21.47-second gZ-36 miss had proposals on only 5 of 43 sampled observations and admitted none. The complete event evidence is in `raw_d5_full_none/fn_waterfall.json`.

## False-positive ledger

Three false positives remain: one on gC-14 and two on gZ-36, lasting 4.03-5.04 seconds. Their maximum proposal confidences are 0.3754-0.4458. Each formed from only one or two admitted observations plus temporal persistence. The exact proposals, association states, and source-model counts are in `raw_d5_full_none/fp_ledger.json`.

## Interpretation

This current raw execution improves over the cached E2 result from 12/13/8 to 14/3/6, but it is not a one-variable causal comparison because cached replay and raw-video inference use different execution paths. The raw result supersedes the earlier pending-parity statement as the best current end-to-end development evidence. Association retention, sparse proposal coverage on gZ-36, sub-second event cadence, and temporal persistence around isolated positives are the next measured bottlenecks.

Artifacts: `raw_d5_full_none/aggregate_metrics.json`, `fn_waterfall.json`, `fp_ledger.json`, `mode_coverage.json`, `runtime.json`, and `artifact_manifest.json`.

## Association bypass upper bound

A pre-registered full-cohort ablation disabled verified association while keeping detector, stride, confidence floor, context rule, temporal configuration, matcher, and action settings fixed. Event metrics were identical on both subjects. Across 1,819 observations, 89 rejection labels changed from `UNVERIFIED_OCCUPANT_ASSOCIATION` to `PHONE_VISIBLE_WITHOUT_HANDHELD_USE`, while admitted observations remained exactly 537.

This rejects association bypass as a recovery mechanism. The current measured limiter is handheld-context geometry or subsequent temporal support. Evidence: `raw_d5_assoc_bypass/experiment_plan.json` and `raw_d5_assoc_bypass/ablation_result.json`.

## Forced handheld-context upper bound

A second pre-registered full-cohort ablation kept verified association enabled and forced every phone proposal to `HANDHELD_USE`. It admitted 42 additional observations and recovered one 21.47-second gZ-36 event, but created two net false positives. Micro counts changed from 14/3/6 to 15/5/5 for TP/FP/FN; precision became 75%, recall 75%, and F1 75%, below the 75.68% baseline.

Forced context is rejected. A valid improvement needs better phone-hand/face evidence and temporal handling rather than removing the context safeguard. Evidence: `raw_d5_handheld_upper_bound/experiment_plan.json` and `raw_d5_handheld_upper_bound/ablation_result.json`.
