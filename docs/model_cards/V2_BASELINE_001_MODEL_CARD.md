# V2_BASELINE_001_MODEL_CARD

## 1. Objective and Architecture
- **Objective:** Detect phone usage and seatbelt status for vehicle occupants to enforce traffic rules.
- **Architecture:** 
  - Phone Detector: YOLO11s (Object Detection)
  - Seatbelt Detector: YOLO11s (Object Detection)
  - Seatbelt Classifier: YOLO11s-cls (Image Classification)

## 2. Datasets
- Phone Detector: `datasets/derived/v2_pretrain_pending_approval/phone_detector`
- Seatbelt Detector: `datasets/derived/v2_pretrain_pending_approval/seatbelt_detector`
- Seatbelt Classifier: `datasets/derived/v2_pretrain_pending_approval/seatbelt_classifier`

## 3. Governance
- **Status:** MODEL_ASSISTED_PENDING_APPROVAL
- **Production Ready:** false
- **Human Verified:** false
- **Integration Status:** CALIBRATED_EVALUATION_CANDIDATE

## 4. Evaluation Metrics

### 4.1. Training-Run Validation Metrics (Source: Ultralytics Training Log)
- **Phone Detector (mAP50 / F1):** 94.48% / 89.34%
- **Seatbelt Detector (mAP50 / F1):** 94.73% / 90.13%
- **Seatbelt Classifier (Top-1):** 80.97%

### 4.2. Canonical Post-Train Validation Metrics (Source: Kaggle Evaluator Version 17)
- **Phone Detector (Val 302 imgs):** Precision 97.0%, Recall 82.7%, mAP50 94.1%, mAP50-95 68.3%
- **Seatbelt Detector (Val 636 imgs):** Precision 92.6%, Recall 87.9%, mAP50 93.9%, mAP50-95 50.3%
- **Seatbelt Classifier (Val 641 imgs, 3-class):** Top-1 Accuracy 81.0%

### 4.3. Canonical Model-Level Frozen Evaluation
*Note: This is a raw model-level capability test executed without explicitly calibrated conf/IOU thresholds or temporal smoothing. It evaluates the raw weights, not the end-to-end system.*
- **Phone Detector (Test 302 imgs, 139 instances):** Precision 90.2%, Recall 86.2%, mAP50 90.5%, mAP50-95 65.5%
- **Seatbelt Detector (Test 614 imgs, 617 instances):** Precision 93.8%, Recall 88.2%, mAP50 92.7%, mAP50-95 47.7%
- **Seatbelt Classifier (Test 621 imgs, 3-class):** Top-1 Accuracy 78.9%

*Note: The canonical frozen test dataset has now been officially consumed (FROZEN_TEST_RUN_COUNT=1). Final system-level Event Evaluation (post-calibration) must utilize a new, un-touched holdout dataset.*

## 5. Behavior Evaluation
EVENT_EVALUATION = PENDING_NEW_UNTOUCHED_HOLDOUT
Final system-level event evaluation has not yet been executed. It requires a newly frozen, untouched sequence holdout with independently reviewed and governed event ground truth. The canonical frozen model test cannot be reused for this purpose.

## 6. Known Failure Modes & Limitations
- **Phone:** Hand near face/ear triggers false positives. Low light reduces recall.
- **Seatbelt:** Dark clothing blends with dark belts causing false negatives.
- **Classifier:** High ambiguity leads to `uncertain_or_occluded`. This class MUST be kept fail-closed (UNKNOWN != UNFASTENED).

## 7. Camera Generalization
CROSS_CAMERA_GENERALIZATION_NOT_PROVEN. This baseline has not been proven against external camera domains.

## 8. Intended Use and Prohibited Interpretation
- **Intended Use:** Review candidate for Driver Monitoring Systems.
- **Prohibited Interpretation:** Do not map `unknown` to `unfastened`. Do not assign PHONE_USE violation to passengers. Seatbelt violations may apply to configured vehicle occupants. Persons outside the vehicle must never trigger either violation. Do not use test metrics to choose models.

## 9. Artifact Hashes
- **Phone Detector SHA256:** `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`
- **Seatbelt Detector SHA256:** `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500`
- **Seatbelt Classifier SHA256:** `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4`
- **Calibration Lock SHA256:** `6b661fdf8bbeec8618d9fd1d3b364ac89bf74398d39f51533fa5d491b0cf9cf2`

## 10. Training Environment
NVIDIA GeForce RTX 3090, 24GB VRAM, Windows 10, PyTorch 2.8.0+cu128.
