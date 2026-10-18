# ROADWATCH V19 — Historical Metric Reproduction & Reconciliation
## 1. Distinction Between Metric Scopes
A critical finding of V19 is establishing the domain boundaries of historical metrics:
- **Detector-level AP (mAP50 = 94.48%):** Bounding box IoU >= 0.5 evaluated on isolated, tightly cropped frames containing prominent phones.
- **Clip-level Action Macro-F1 (56.39%):** Frame/clip classification accuracy across 3 coarse classes (SAFE, PHONE, OTHER) on development LOSO splits.
- **Event-level F1 (65.31%):** End-to-end continuous temporal event matching on the 3-subject cohort (29 GT events across 22.06 minutes).

## 2. Derivation of the 29-Event Cohort
The historical baseline metrics (TP=16, FP=4, FN=13, Precision=80.00%, Recall=55.17%, F1=65.31%, FA/min=0.1813) originated from `reports/ai85_next/METRICS_COMPARISON.md` under `SN_PHONE_INHERITED_POLICY`:
- **Subject `gB-9`:** 9 GT events (0 TP, 0 FP, 9 FN under baseline full-frame detector).
- **Subject `gZ-36`:** 11 GT events.
- **Subject `gC-14`:** 9 GT events.
- Total: 29 GT events.
