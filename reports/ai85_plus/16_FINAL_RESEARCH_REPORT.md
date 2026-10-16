# Roadwatch AI85+ V8 final research report

**Status: `INDEPENDENT_AI85_NOT_ESTABLISHED`. No new production-approved or selected candidate.**

Software validation: 552 full-suite tests passed, with 0 failures, 0 errors and 0 skips. Compile, critical Ruff, frontend build and backend/inference smoke passed.

Delivered working research code and measurements: safe root resolution, all-checkpoint inventory, original-video dense/adaptive sampling, bounded confirmation, ROI box restoration, missing-view abstention, 91 frozen detector spatial probes, three fresh seatbelt inference runs, original FN traces, system smoke and runtime benchmark.

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

The temporal hypothesis is partly falsified: dense sampling covers all four subsecond events but recognizes none under the frozen model/protocol. It recovers one long event while adding enough false alarms to reduce F1. Representation confidence and phone localization remain stronger bottlenecks than sample coverage alone. Simple motion triggers and confirmation do not improve event F1 here.
The controlled historical binary branch remains a useful development comparator (80% P, 55.17% R, 65.31% F1), not a new approved model. Direct coarse SAFE 70.81% is a separate result. Seatbelt metrics are in report 11 and cannot certify the 85% target with pending-approval labels.

Targets were not established independently. Detector governed AP is blocked by approved bbox GT; new training is blocked by confirmed labels and independent splits; independent final evaluation is blocked by a new lawful holdout. Existing protected subjects were not reused. The source/test path defects were resolved, while historical manifests retain their original lineage and paths.
Next admissible work is reviewed phone boxes/hard negatives and complete three-class seatbelt labels with real subject/camera grouping, then locked validation for representation and ROI experiments. No further threshold search on these three subjects was performed.

Evidence: reports 01–15, checkpoint_metadata.json, metrics_comparison.json, temporal predictions/masks/matches, seatbelt prediction CSVs, source/model/input hashes, artifact validation and [reproduction commands](reproduce.ps1). [Primary research references](references.md).

Private Kaggle job [longngoc11/roadwatch-ai85-plus-frozen-gpu-v8](https://www.kaggle.com/code/longngoc11/roadwatch-ai85-plus-frozen-gpu-v8): COMPLETE. It runs frozen phone inference on 64 legacy validation images already in the account. No DMD media, holdouts or credentials were uploaded.

| GPU | Variant | Images | Detector-call FPS | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- |
| Tesla T4 | original_960 | 64 | 21.44 | 34.12 | 76.3 |
| Tesla T4 | resolution_640 | 64 | 38.47 | 15.29 | 67.08 |
| Tesla T4 | TTA_640 | 64 | 19.89 | 35.12 | 128.8 |
| Tesla T4 | resolution_1024 | 64 | 28.84 | 27.46 | 70.37 |
| Tesla T4 | agnostic_640 | 64 | 66.24 | 14.88 | 17.02 |

GPU and local CPU probes use different image sets; their throughput is not a controlled hardware speedup comparison. GPU detector-call FPS is not full-system FPS, and no AP/accuracy target was measured on Kaggle.
