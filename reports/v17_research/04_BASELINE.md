# 04. V16 Baseline Metrics & Benchmark State

## 1. Controlled Development Metrics (V16 Verified Baseline)
* **Phone Event Detection (Controlled 29-Event Cohort across 22.06 minutes):**
  * True Positives (TP): **16**
  * False Positives (FP): **4**
  * False Negatives (FN): **13**
  * Precision: **80.00%**
  * Recall: **55.17%**
  * Event F1: **65.31%**
  * False Alarm Rate: **0.1813 FA/min** (approx. 1 false alert every 5.5 minutes)
* **Action Recognition (E3 Development Model on 2,625 clips):**
  * Macro-F1: **56.39%**
  * Balanced Accuracy: **56.81%**
  * SAFE_DRIVING Recall: **60.38%**
  * PHONE_USE Recall: **66.78%**
  * OTHER_ACTIVITY Recall: **43.26%**
  * Worst-Subject SAFE Recall (Subject gZ-36): **1.66%** (Severe subject collapse)
* **Seatbelt State Classification (S2 Model on Development Cohort):**
  * Macro-F1: **58.52%**
  * FASTENED Recall: **4.00%**
  * UNCERTAIN Recall: **4.00%**
  * False-Unfastened Rate: **8.00%**
  * Abstention Rate: **5.46%**

## 2. Distance to Target
| Metric | V16 Baseline | V17 Target | Gap |
|---|---|---|---|
| Phone Event Precision | 80.00% | >85.00% | +5.00% |
| Phone Event Recall | 55.17% | >85.00% | +29.83% |
| Phone Event F1 | 65.31% | >85.00% | +19.69% |
| False Alarm Rate | 0.1813 | <=0.1500 | -0.0313 FA/min |
| Action Macro-F1 | 56.39% | >85.00% | +28.61% |
| SAFE_DRIVING Recall | 60.38% | >85.00% | +24.62% |
| Seatbelt Macro-F1 | 58.52% | >85.00% | +26.48% |
