# V2 Seatbelt Classifier Calibration

## 1. Overview
- **Component:** Seatbelt Classifier (YOLO11s-cls)
- **Model SHA:** `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4`
- **Calibration Split:** `datasets/derived/v2_pretrain_pending_approval/seatbelt_classifier/val` (641 images)

## 2. Reconciled Raw Confusion Matrix & Base Metrics
*(Note: Previous reports suffered from an axis transposition where True/Predicted and Precision/Recall were mathematically swapped. This has been fully reconciled based on raw inference logs.)*

**Raw Evaluation Matrix (Before Reject Policy):**

| True \ Predicted      | Fastened | Unfastened | Uncertain | Total |
|-----------------------|----------|------------|-----------|-------|
| **Fastened**          | 300      | 49         | 1         | 350   |
| **Unfastened**        | 47       | 218        | 1         | 266   |
| **Uncertain**         | 14       | 10         | 1         | 25    |

**Raw Class-Level Performance:**
- **Fastened:** Precision 83.10%, Recall 85.71%, F1 84.39%
- **Unfastened:** Precision 78.70%, Recall 81.95%, F1 80.29%
- **Uncertain/Occluded:** Precision 33.33%, Recall 4.00%, F1 7.14%

**Overall Raw Metrics:**
- **Top-1 Accuracy:** 80.97% (519/641)
- **Macro F1 (3-class):** 57.27%

## 3. Reject Policy Threshold Sweep
To minimize the risk of raising false positive unfastened alarms, a full threshold sweep was performed across raw classification probabilities (`seatbelt_classifier_threshold_sweep.csv`). 

At the raw operating point, there are **49** false unfastened violations (`True Fastened -> Pred Unfastened`).
Since YOLO11s-cls assigns high base confidence to its argmax predictions (minimum `0.504`), sweeping the fastened threshold up to `0.50` yields identical results, whereas the `Unfastened >= 0.60` threshold actively suppresses marginal violation predictions.

**Selected Operating Point (Fastened >= 0.50 / Unfastened >= 0.60):**
- **Coverage:** 95.16% (Only 4.8% of images rejected)
- **Macro F1 (Main Classes):** 81.61%
- **False Unfastened Count:** Reduced from 49 down to 36 (a ~26.5% reduction in false alarms).
- **False Fastened Count:** 47

## 4. Post-Policy Confusion Matrix

After explicitly forcing classifications with `max_prob < threshold` into the `uncertain_or_occluded` bucket, the matrix becomes:

| True \ Predicted      | Fastened | Unfastened | Uncertain | Total |
|-----------------------|----------|------------|-----------|-------|
| **Fastened**          | 300      | 36         | 14        | 350   |
| **Unfastened**        | 47       | 203        | 16        | 266   |
| **Uncertain**         | 14       | 10         | 1         | 25    |

**Post-Policy Class-Level Performance:**
- **Fastened:** Precision 83.10%, Recall 85.71%, F1 84.39%
- **Unfastened:** Precision 81.53%, Recall 76.32%, F1 78.83%
- **Uncertain/Occluded:** Precision 3.23%, Recall 4.00%, F1 3.57%

## 5. Limitations & Evidence Lock
- **Uncertain Support Limitation:** The canonical validation set contains only 25 True `uncertain_or_occluded` samples, which limits confidence in the rejection metrics.
- **Fail-Closed Constraint:** Despite low F1 on `uncertain_or_occluded`, the policy successfully acts as an active noise sink, capturing 30 uncertain predictions, thereby shielding the final event engine from false violations.

**Lock Conclusion:** `VALIDATION_CALIBRATION` is completely reconciled. The threshold matrix and metrics are scientifically verified and permanently locked.
