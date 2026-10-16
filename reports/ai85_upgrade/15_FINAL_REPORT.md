# AI85 upgrade: actual results and remaining work

The requested audit, code changes, controlled CPU training, detector probe and evaluation were executed. The 85% target is not established. No candidate is production-approved.

| Experiment | TP | FP | FN | Precision | Recall | F1 | FA/min | SAFE | Macro-F1 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| V7_A_V6_BASELINE_ACTION | 15 | 5 | 14 | 75.00% | 51.72% | 61.22% | 0.2266 | 12.57% | 34.23% |
| V7_A_V6_BASELINE_FUSION | 16 | 5 | 13 | 76.19% | 55.17% | 64.00% | 0.2266 | 12.57% | 34.23% |
| BASELINE_NEGATIVE_FUSION | 16 | 5 | 13 | 76.19% | 55.17% | 64.00% | 0.2266 | 12.57% | 34.23% |
| MEAN_FINETUNE_ACTION | 15 | 6 | 14 | 71.43% | 51.72% | 60.00% | 0.2719 | 12.13% | 34.16% |
| MEAN_FINETUNE_FUSION | 16 | 6 | 13 | 72.73% | 55.17% | 62.75% | 0.2719 | 12.13% | 34.16% |
| MEAN_FINETUNE_NEGATIVE_FUSION | 16 | 6 | 13 | 72.73% | 55.17% | 62.75% | 0.2719 | 12.13% | 34.16% |
| ATTENTION_FINETUNE_ACTION | 15 | 6 | 14 | 71.43% | 51.72% | 60.00% | 0.2719 | 12.13% | 34.11% |
| ATTENTION_FINETUNE_FUSION | 16 | 6 | 13 | 72.73% | 55.17% | 62.75% | 0.2719 | 12.13% | 34.11% |
| ATTENTION_FINETUNE_NEGATIVE_FUSION | 16 | 6 | 13 | 72.73% | 55.17% | 62.75% | 0.2719 | 12.13% | 34.11% |
| NATIVE112_FINETUNE_ACTION | 13 | 7 | 16 | 65.00% | 44.83% | 53.06% | 0.3173 | 19.01% | 40.21% |
| NATIVE112_FINETUNE_FUSION | 14 | 7 | 15 | 66.67% | 48.28% | 56.00% | 0.3173 | 19.01% | 40.21% |
| NATIVE112_FINETUNE_NEGATIVE_FUSION | 14 | 7 | 15 | 66.67% | 48.28% | 56.00% | 0.3173 | 19.01% | 40.21% |

Highest observed development F1 in this run: `V7_A_V6_BASELINE_FUSION`, Precision 76.19%, Recall 55.17%, F1 64.00%. This is descriptive and does not select a production model.

Implemented checkpoint-compatible temporal attention, deterministic fine-tuning with explicit subject isolation, direct original-resolution ROI embedding extraction, negative-evidence fusion, and an error audit that includes duplicate false positives. All 13 FN cases have original ROI evidence. Failed experiments are retained.

The remaining evidence gaps are new independent subjects, representative approved UNCERTAIN seatbelt training labels, verified phone boxes/association evidence, and stronger cross-subject representations. Repeating threshold searches on these three subjects would not resolve those gaps. See `14_TEST_RESULTS.md` for actual test outcomes and unresolved inherited failures.

Reproduce with Python 3.11 from the repository root; output directories must be fresh because existing fold checkpoints are protected from overwrite:

```powershell
py -3.11 -m training.dmd.run_ai85_upgrade --stage audit
py -3.11 -m training.dmd.run_ai85_upgrade --stage train
py -3.11 -m training.dmd.ai85_detector_probe
py -3.11 -m training.dmd.ai85_native_embeddings --subject gB-9
py -3.11 -m training.dmd.ai85_native_embeddings --subject gZ-36
py -3.11 -m training.dmd.ai85_native_embeddings --subject gC-14
py -3.11 -m training.dmd.run_ai85_upgrade --stage train --config models/model_config_ai85_native112.yaml --output reports/ai85_upgrade/native112
py -3.11 -m training.evaluate_seatbelt_v6 --output reports/ai85_upgrade/seatbelt_metrics.json
py -3.11 -m training.benchmark_v6_runtime --frames 12 --output reports/ai85_upgrade/runtime_benchmark.json
py -3.11 -m training.dmd.verify_ai85
py -3.11 -m training.dmd.render_ai85_reports
```
