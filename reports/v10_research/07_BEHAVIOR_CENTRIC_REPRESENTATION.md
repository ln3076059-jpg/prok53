# V10 behavior-centric representation probe

Evidence level: **LEVEL 2 - development ablation**. All folds are subject-disjoint LOSO across
`gB-9`, `gC-14`, and `gZ-36`; these subjects have been used repeatedly and are not independent.

| Representation | Dim | Macro-F1 | Balanced accuracy | SAFE recall | Action silhouette | Subject silhouette | Subject/action |
| --- | --- | --- | --- | --- | --- | --- | --- |
| D0_FROZEN_RESNET_FULL_ROI | 512 | 17.56% | 32.35% | 6.82% | 0.0278 | 0.4014 | 14.4547 |
| D1_BEHAVIOR_TWO_REGION_HOG | 648 | 45.17% | 48.30% | 35.56% | 0.0432 | 0.4077 | 9.4376 |
| D2_TEMPORAL_RESIDUAL | 512 | 51.54% | 53.57% | 44.27% | 0.0046 | 0.0895 | 19.391 |
| D5_BEHAVIOR_HOG_PLUS_MOTION | 1160 | 48.63% | 56.04% | 43.25% | 0.0245 | 0.2531 | 10.3495 |

| Representation | Held-out subject | Macro-F1 | SAFE recall | Clips |
| --- | --- | --- | --- | --- |
| D0_FROZEN_RESNET_FULL_ROI | gB-9 | 16.22% | 0.00% | 808 |
| D0_FROZEN_RESNET_FULL_ROI | gC-14 | 14.63% | 3.85% | 794 |
| D0_FROZEN_RESNET_FULL_ROI | gZ-36 | 19.61% | 16.18% | 1023 |
| D1_BEHAVIOR_TWO_REGION_HOG | gB-9 | 66.56% | 95.42% | 808 |
| D1_BEHAVIOR_TWO_REGION_HOG | gC-14 | 40.53% | 7.21% | 794 |
| D1_BEHAVIOR_TWO_REGION_HOG | gZ-36 | 9.88% | 0.41% | 1023 |
| D2_TEMPORAL_RESIDUAL | gB-9 | 54.71% | 70.42% | 808 |
| D2_TEMPORAL_RESIDUAL | gC-14 | 50.48% | 63.46% | 794 |
| D2_TEMPORAL_RESIDUAL | gZ-36 | 37.64% | 1.66% | 1023 |
| D5_BEHAVIOR_HOG_PLUS_MOTION | gB-9 | 56.57% | 91.25% | 808 |
| D5_BEHAVIOR_HOG_PLUS_MOTION | gC-14 | 68.32% | 36.54% | 794 |
| D5_BEHAVIOR_HOG_PLUS_MOTION | gZ-36 | 12.39% | 1.24% | 1023 |

The pre-written pooled selection rule requires higher Macro-F1 than D0 without reducing pooled SAFE recall.
It selects **D2_TEMPORAL_RESIDUAL** for follow-up. Hypothesis outcome: **PARTIALLY_SUPPORTED**;
model-selection decision: **REJECT_FOR_PROMOTION**.

The representation hypothesis is only partly supported. D2 substantially reduces subject silhouette and
improves pooled clip metrics, but its action silhouette is lower than D0 and its worst held-out subject is
`gZ-36`, where SAFE recall is 1.66%. Every representation remains below the 85% research targets. The pooled
rule is retained as originally specified; the worst-subject safety result separately blocks promotion.

D0 reuses frozen ResNet-18 full-driver-ROI embeddings. D1 uses fixed overlapping upper/lower HOG regions;
D2 uses adjacent-frame residual maps; D5 concatenates D1 and D2. The same balanced linear head, seed and
LOSO folds are used throughout. No backbone, detector, event threshold or matcher was changed.

This test does not measure Event F1. The RGB cache is 64x64, D2 includes future-frame evidence, dataset
rights remain unaccepted, and no independent holdout exists. The candidate can guide the next controlled
video experiment but cannot be promoted to production.
