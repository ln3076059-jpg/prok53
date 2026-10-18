# 14. Event Metrics Reconciliation & Target Evaluation

## 1. Reconciled Metrics Table
| Metric | Baseline V16 | V17 Current Status | Target AI85 | Evaluation Status |
|---|---|---|---|---|
| True Positives (TP) | 16 | 16 (Delta = 0) | >25 | FROZEN_DEVELOPMENT |
| False Positives (FP) | 4 | 4 (Delta = 0) | <=3 | FROZEN_DEVELOPMENT |
| False Negatives (FN) | 13 | 13 (Delta = 0) | <=4 | FROZEN_DEVELOPMENT |
| Precision | 80.00% | 80.00% | >85.00% | TARGET_NOT_MET |
| Recall | 55.17% | 55.17% | >85.00% | TARGET_NOT_MET |
| Event F1-Score | 65.31% | 65.31% | >85.00% | TARGET_NOT_MET |
| False Alarms / min | 0.1813 | 0.1813 | <=0.1500 | TARGET_NOT_MET |
| Action Macro-F1 | 56.39% | 56.39% | >85.00% | TARGET_NOT_MET |
| SAFE_DRIVING Recall | 60.38% | 60.38% | >85.00% | TARGET_NOT_MET |
| Seatbelt Macro-F1 | 58.52% | 58.52% | >85.00% | TARGET_NOT_MET |

## 2. Scientific Integrity Statement
No prediction outputs were artificially modified. All baseline metrics remain exactly matched to audited truth until newly authorized training data is ingested.
