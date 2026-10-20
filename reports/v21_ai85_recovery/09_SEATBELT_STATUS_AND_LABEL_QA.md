# ROADWATCH V21 — Engineering Report 09
# Seatbelt AI85 Status, Classifier Diagnostics, and Label QA Workflow Audit

**Author:** Senior Deep Learning Engineer & Data Engineering Lead  
**Date:** 2026-10-20  
**Status:** COMPLETE AUDIT & QA WORKFLOW FORMULATION  

---

## 1. Problem Statement: The 85% Seatbelt Objective

The project objective specifies:
- **Seatbelt Macro-F1:** $>85.0\%$
- **Current Reported Macro-F1:** $57.28\%\text{--}58.52\%$
- **Current Reported Accuracy:** $80.97\%$

A major deficiency in earlier assessments was confusing the occupant torso detector's high AP50 ($94.73\%$) with the 3-class seatbelt classifier's Macro-F1. While detecting the torso is straightforward, correctly classifying seatbelt status across all three classes—especially under partial occlusion and dark clothing—remains far below the 85% target.

---

## 2. Checkpoint & Validation Split Audit

We audited the existing active seatbelt artifacts in `models/active/v2/`:
- `seatbelt_detector.pt`: YOLOv8 torso detector (640x640, SHA-256: `c8d197...`)
- `seatbelt_classifier.pt`: ResNet18 torso classifier (224x224, 3 classes, SHA-256: `318b87...`)
- Validation evaluation file: `reports/v6_completion/v6_seatbelt_evaluation.json` (641 verified samples).

### Baseline Confusion Matrix (641 Samples)

```
Predicted Class ->      seatbelt_fastened    seatbelt_unfastened    uncertain_or_occluded    Total (Support)
True Class:
seatbelt_fastened              300                    49                      1                   350
seatbelt_unfastened             47                   218                      1                   266
uncertain_or_occluded           14                    10                      1                    25
Total                          361                   277                      3                   641
```

### Detailed Per-Class Breakdown

| Class Name | Support | Precision | Recall | F1-Score | AI85 Target | Status |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **`seatbelt_fastened`** | 350 | 83.10% | **85.71%** | **84.39%** | F1 $>85.0\%$ | **Near Target (-0.61%)** |
| **`seatbelt_unfastened`** | 266 | 78.70% | **81.95%** | **80.29%** | F1 $>85.0\%$ | **Sub-target (-4.71%)** |
| **`uncertain_or_occluded`** | 25 | 33.33% | **4.00%** | **7.14%** | F1 $>85.0\%$ | **CRITICAL COLLAPSE (-77.86%)** |
| **MACRO AVERAGE** | **641** | **65.04%** | **57.22%** | **57.28%** | **Macro-F1 >85.0%** | **FAIL (57.28%)** |

### The Core Diagnostic Finding
The model almost **never predicts `uncertain_or_occluded`** (only 3 total predictions across 641 samples, yielding a recall of only $4.00\%$). Because Macro-F1 equally weights all three classes:
$$\text{Macro-F1} = \frac{84.39\% + 80.29\% + 7.14\%}{3} = 57.27\%$$
The collapse of the `uncertain_or_occluded` class single-handedly drags Macro-F1 down from $\sim 82\%$ to $57.28\%$.

---

## 3. Root Cause Analysis: Extreme Class Imbalance & Threshold Starvation

1. **Extreme Class Imbalance in Training Data:**
   The training split contained hundreds of fastened and unfastened examples, but fewer than 30 uncertain crops. The cross-entropy loss penalized errors on the dominant classes, driving the model's logits for the uncertain class into severe negative bias.
2. **Post-Policy Threshold Inversion:**
   When post-hoc threshold rules were applied in V6 (`fastened_min_confidence = 0.50`, `unfastened_min_confidence = 0.60`), 28 borderline predictions were forced into `uncertain_or_occluded`. However, because the uncalibrated probabilities were already skewed, precision on the uncertain class dropped from $33.33\%$ down to $3.23\%$, lowering Macro-F1 further to $55.60\%$.

---

## 4. Audit of the 500 Pending Review Samples

We investigated `reports/v7_optimization/seatbelt_uncertain_review_queue.csv` containing 26 high-uncertainty validation crops and 500 candidate training samples:

### Data Governance & Integrity Findings
- **Local Pixel Availability:** For all 26 queued rows in the review CSV, `roi_available_locally = False`. The high-resolution cropped image files are archived in cold storage or remote dataset bundles.
- **Scientific Prohibition on Auto-Approval:** The guidelines explicitly state:
  > *"Do not auto-approve labels. A review queue must require explicit human visual verification."*
- **Integrity Compliance:** Because ROI pixels could not be visually reviewed locally by a human annotator during this turn, **zero labels were fabricated or auto-approved**, and **no model weights were overwritten**.

---

## 5. Roadmap to AI85 Seatbelt Performance

To achieve Macro-F1 $>85.0\%$ scientifically without falsifying metrics:

1. **Class-Balanced Focal Loss / Cost-Sensitive Training:**
   Train the ResNet torso head using focal loss ($\gamma = 2.0$, $\alpha = [0.25, 0.35, 0.40]$) to heavily penalize misclassifications of the `uncertain_or_occluded` class.
2. **Torso Keypoint Alignment via Pose:**
   Crop the torso ROI dynamically using the 4 pose keypoints (`left_shoulder`, `right_shoulder`, `left_hip`, `right_hip`). Standardizing the collarbone-to-hip diagonal eliminates non-torso background distractors.
3. **Execution of the Human-in-the-Loop QA Review:**
   Once local pixels are unpacked from remote storage, execute human-verified QA on the 500 pending samples to expand the `uncertain` class support from 25 to $>200$ verified instances.
