# Evaluation

Run validation first and calibrate thresholds from validation only. Create `reports/model_lock.json` with weights, dataset, split, code, versions, and thresholds. `training/evaluate.py --split test` refuses to run before that lock. Frozen test runs once. Report per-class precision, recall, F1, AP50, and AP50-95 plus macro F1 and aggregate mAP.

System event metrics remain `NOT_RUN` until humans independently annotate licensed validation/test videos.

