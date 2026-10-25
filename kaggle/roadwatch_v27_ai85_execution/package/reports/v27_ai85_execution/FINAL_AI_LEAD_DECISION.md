# V27 final AI lead decision

Decision date: 2026-10-21.

## Decision

`DEVELOPMENT_TARGET_NOT_MET`

The best current end-to-end development result is the completed raw D5 run: TP/FP/FN = 14/3/6, precision = 0.8235, recall = 0.7000, and F1 = 0.7568. The mandatory condition precision >0.85, recall >0.85, and F1 >0.85 is not met.

## Accepted engineering changes

- Candidate selection now precedes action inference, with one candidate identity carried through action and event output.
- Requested deep modes fail closed instead of silently becoming rule-only.
- Centered action scores are aligned to their exact target record in cached replay; raw online inference rejects this checkpoint contract until it has an equivalent delayed-output queue.
- Detector, pose, and action inference accept the same device selection and export action provenance.
- E4/E5 were replayed without fitting. E4 improved from one extra FP to parity with E2; E5 remains worse than baseline.
- The relevant V22-V27, temporal, governance, and pose suite passed 103/103 tests.

## Kaggle result

`V27_GPU_PREFLIGHT_PASSED`

Private Kaggle kernel version 3 completed on a Tesla T4. Specialist, auxiliary, and pose all reported `cuda:0`; median call latency was 19.68 ms, 11.90 ms, and 11.63 ms respectively over 32 existing consumed validation images. The run performed no training, used no DMD media, and computed no event accuracy metric.

`BLOCKED_NO_AUTHORIZED_DMD_MEDIA_ON_KAGGLE`

The current Kaggle account has no private dataset containing the authorized gC-14/gZ-36 source videos, and the source record permits code-only external upload. Raw D5 parity therefore cannot be run on Kaggle under the current data permission. This blocker must not be bypassed by uploading DMD video or labels.

## Remaining scientific bottleneck

The best run still needs at least six additional TP and ten fewer FP to reach a feasible 18/3/2 outcome on the 20-event development cohort. gZ-36 contributes 5/10/6, so proposal recall, association retention, short-event sampling, and duplicate event formation dominate the remaining gap. Action fusion alone did not improve recall.

Full raw D5 parity and its FN waterfall/FP ledger are now complete. Of six FN, four are sub-second, three are dominated by association UNKNOWN rejection, one has sparse proposals plus rejection, one has no proposal, and one has only one admitted observation. Detector bbox recall and role-association accuracy still require independent governed component labels; until those labels exist, those formal component metrics remain `NOT_ESTABLISHED`.

Evidence: [execution report](EXECUTION_REPORT.md), [metrics](metrics_comparison.json), [Kaggle receipt](kaggle_execution_receipt.json), [Kaggle result](kaggle_gpu_preflight_v3/v27_gpu_preflight.json), and [tests](test_results.json).

## Additional Kaggle component evidence

Private kernel `longngoc11/roadwatch-v27-phone-component-eval` version 2 completed a frozen-checkpoint diagnostic on 1,782 bootstrap validation images. Input size 960 produced precision 0.9395, recall 0.9039, F1 0.9214, mAP50 0.9456, and 19.83 ms inference per image on a Tesla T4. The 640 and 1280 variants produced F1 0.8499 and 0.8031 respectively.

This does not change the final decision. The bootstrap labels are model proposals rather than governed final ground truth, no independent test split was accessed, and this bbox component task is different from DMD event detection. The valid system-level result remains precision 0.48, recall 0.60, and F1 0.5333. Evidence: [component receipt](kaggle_phone_component_receipt.json) and [raw component result](kaggle_phone_component_v2/v27_phone_component_eval.json).

The corresponding code, audit, aggregate result, and reports were published in private code-only dataset `longngoc11/roadwatch-v27-ai85-execution` version 6. Its 42 packaged files match the manifest and contain no DMD media or credentials.

## Completed raw D5 evidence

Local raw D5 processed both full development videos: gC-14 = 7/1/2 and gZ-36 = 7/2/4 for TP/FP/FN. Micro precision is 0.8235, recall 0.7000, and F1 0.7568. This supersedes the earlier pending-parity statement, while remaining development-only and below the mandatory gate. Evidence: [raw report](RAW_D5_REPORT.md), [aggregate metrics](raw_d5_full_none/aggregate_metrics.json), [FN waterfall](raw_d5_full_none/fn_waterfall.json), and [FP ledger](raw_d5_full_none/fp_ledger.json).

Two pre-registered diagnostic upper bounds were also completed. Bypassing verified association changed no admissions and no event metrics. Forcing handheld context recovered one TP but added two net FP, producing 15/5/5 and F1 0.7500. Both bypasses are rejected; the 14/3/6 raw baseline remains the best valid current result.
