# V27 AI85 correctness execution report

Generated 2026-10-21T02:18:31.840496+00:00. Scope: cached replay correctness ablation on the existing two-subject development cohort. No parameter, threshold, label, or model fitting was performed in this execution.

## Result

| Run | TP | FP | FN | Precision | Recall | F1 | FA/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| E2 baseline | 12 | 13 | 8 | 48.00% | 60.00% | 53.33% | 0.8513 |
| E4 aligned deep | 12 | 13 | 8 | 48.00% | 60.00% | 53.33% | 0.8513 |
| E5 aligned hybrid | 12 | 14 | 8 | 46.15% | 60.00% | 52.17% | 0.9168 |

The mandatory gate is **NOT_MET**. E4 no longer adds the extra false positive seen in V26, but the best measured result remains E2/E4 at F1 53.33%. Reaching all three metrics above 85% on 20 GT events requires at least 18 TP and at most 3 FP; the current best has 12 TP and 13 FP.

## Correctness invariants

| Run | Valid centered scores | Wrong target frame | Target time error >1 ms | Emitted before target | Candidate scored | Candidate ID mismatch | Declared latency range |
|---|---:|---:|---:|---:|---:|---:|---|
| E4 | 1803 | 0 | 0 | 0 | 563 | 0 | 2.016083-2.016601s |
| E5 | 1803 | 0 | 0 | 0 | 563 | 0 | 2.016083-2.016601s |

The old V26 trace attached every centered score about 2.016 seconds late to the currently processed observation. V27 maps the score back to its exact target frame, uses that frame's pose and selected candidate, and records availability separately. The raw online runner now rejects centered checkpoints because it has no target-aligned delayed-output queue.

## Verification

The focused V22-V27, temporal business-rule, evaluation-governance, and pose-feature suite passed 103/103 tests. The machine-readable result is in `test_results.json`; individual cases and timings are in `junit_report.xml`.


## Kaggle GPU preflight

Kaggle kernel version 3 completed on **Tesla T4**. Device provenance is specialist=cuda:0, auxiliary=cuda:0, and pose=cuda:0. Median call latency was 19.68 ms, 11.90 ms, and 11.63 ms respectively on 32 existing private validation images.

This run performed no training, used no DMD media, and computed no event metric. Raw D5 remains `BLOCKED_NO_AUTHORIZED_DMD_MEDIA_ON_KAGGLE` because the account has no authorized DMD media dataset. The signed local receipt is `kaggle_execution_receipt.json`.


## Remaining measured bottleneck

The best run has gC-14 = 7/3/2 and gZ-36 = 5/10/6 for TP/FP/FN. The dominant gap is proposal and association quality on gZ-36, plus short-event sampling; action fusion alone cannot recover the missing six true events or remove ten false events needed to pass the strict gate. The next valid experiment is a pre-registered raw D5 detector/cadence study with component-level bbox and association labels, followed by source-aware calibration on development data and one frozen independent evaluation.

## Kaggle phone-detector component diagnostic

Private Kaggle kernel version 2 evaluated the frozen phone checkpoint on the existing `roadwatch-mc-bootstrap-6500-v1` validation split (1,782 images, 181 positive images, 229 phone boxes) on a Tesla T4. It did not train, access a test split, or use DMD media.

| Input size | Precision | Recall | F1 from P/R | mAP50 | mAP50-95 | Inference ms/image | Peak CUDA |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 640 | 88.09% | 82.10% | 84.99% | 89.74% | 64.33% | 7.99 | 1.53 GiB |
| 960 | 93.95% | 90.39% | 92.14% | 94.56% | 74.97% | 19.83 | 3.24 GiB |
| 1280 | 82.57% | 78.17% | 80.31% | 85.27% | 61.89% | 38.90 | 5.61 GiB |

The best observed resolution by diagnostic F1 is 960. The non-monotonic 1280 result shows that increasing input size is not a safe substitute for validation. These are provisional component metrics because the labels are explicitly `PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA`; they are ineligible for model selection, production claims, or evidence that event-level precision/recall/F1 exceeds 85%. The system decision therefore remains `DEVELOPMENT_TARGET_NOT_MET`. Evidence is recorded in `kaggle_phone_component_receipt.json`.

The private code-only dataset `longngoc11/roadwatch-v27-ai85-execution` was updated as version 6. Its latest package contains 42 manifest-listed files, including the component kernel, raw D5 audit code, aggregate result, ablation auditors, and decision reports. Remote listing matched the package; no `__pycache__`, bytecode, credential, DMD media, annotation, embedding, checkpoint, or raw prediction was included. The verification record is `kaggle_package_v6_receipt.json`.

## Full local raw D5 parity

The previously pending raw D5 run is now complete on all 27,267 source frames from gC-14 and gZ-36. At stride 15 it processed 1,819 samples in 760.09 seconds on CPU with D5 detection, verified association, no action model, and no action gating.

| Run | TP | FP | FN | Precision | Recall | F1 | FA/min |
|---|---:|---:|---:|---:|---:|---:|---:|
| Cached E2 | 12 | 13 | 8 | 48.00% | 60.00% | 53.33% | 0.8513 |
| Raw D5 current | 14 | 3 | 6 | 82.35% | 70.00% | 75.68% | 0.1965 |

The raw result is the best current end-to-end development evidence, but the delta against cached E2 is not a single-variable estimate because the execution paths differ. The >85% gate remains unmet. Four of six FN events are shorter than one second; three FN are dominated by association UNKNOWN rejection, one by sparse proposals plus rejection, one has no proposal, and one has only one admitted observation. Full evidence is in `RAW_D5_REPORT.md` and `raw_d5_full_none/`.

Two pre-registered upper bounds refined the root cause. Disabling verified association changed 89 rejection labels but zero admissions and zero event metrics. Forcing `HANDHELD_USE` admitted 42 more observations and recovered one long gZ-36 event, but added two net FP; its 15/5/5 result has F1 75.00%, below the 14/3/6 baseline at 75.68%. Both bypasses are rejected. The next valid work is governed phone-hand/face evidence, proposal coverage, and cadence for four sub-second FN events.
