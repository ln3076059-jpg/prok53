# V2 Seatbelt Classifier Calibration

## 1. Overview
- **Component:** Seatbelt Classifier (YOLO11s-cls)
- **Model SHA:** `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4`
- **Calibration Split:** `datasets/derived/v2_pretrain_pending_approval/seatbelt_classifier/val` (641 images)

## 2. Confusion Matrix & Base Metrics
The raw evaluation matrix on 641 images before applying any reject policy:

| True \ Predicted      | Fastened | Unfastened | Uncertain |
|-----------------------|----------|------------|-----------|
| **Fastened**          | 300      | 47         | 14        |
| **Unfastened**        | 49       | 218        | 10        |
| **Uncertain**         | 1        | 1          | 1         |

**Class-Level Performance:**
- **Fastened:** Precision 85.71%, Recall 83.10%, F1 84.39%
- **Unfastened:** Precision 81.95%, Recall 78.70%, F1 80.29%
- **Uncertain/Occluded:** Precision 4.00%, Recall 33.33%, F1 7.14%

**Overall:**
- **Top-1 Accuracy:** 81.0%
- **Macro F1:** 57.27%
- **Weighted F1:** 82.23%

## 3. Reject Policy (Threshold Sweep & Evidence)
Currently, 47 fastened belts are falsely classified as unfastened (false positive violations). To minimize the risk of raising false positive unfastened alarms, a full threshold sweep was performed across raw classification probabilities (see `seatbelt_classifier_threshold_sweep.csv`).

**Key Operating Point Evidence (Fastened 0.50 / Unfastened 0.60):**
- **Coverage:** 95.16% (Only 4.8% of images rejected)
- **Macro F1 (Main Classes):** 81.61%
- **False Unfastened Count:** Reduced from 47 down to 36 (a ~23.4% reduction in false alarms).
- **False Fastened Count:** 47

**Policy Rule:**
- **Unfastened Confidence Threshold:** `0.60`
- **Fastened Confidence Threshold:** `0.50`

If the maximum probability predicted by the classifier falls below the specific threshold for its class, the system explicitly overrides the classification to `uncertain_or_occluded`. This fail-closed approach empirically reduces confident false unfastened violations while maintaining excellent coverage.
