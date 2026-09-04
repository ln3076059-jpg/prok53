# V2_BASELINE_001_MODEL_CARD

## 1. Objective and Architecture
- **Objective:** Detect phone usage and seatbelt status for vehicle occupants to enforce traffic rules.
- **Architecture:** 
  - Phone Detector: YOLO11s (Object Detection)
  - Seatbelt Detector: YOLO11s (Object Detection)
  - Seatbelt Classifier: YOLO11s-cls (Image Classification)

## 2. Datasets
- Phone Detector: `datasets/derived/phone_bootstrap_v2`
- Seatbelt Detector: `datasets/derived/seatbelt_v2_balanced`
- Seatbelt Classifier: `datasets/derived/seatbelt_classifier_v2`

## 3. Governance
- **Status:** MODEL_ASSISTED_PENDING_APPROVAL
- **Production Ready:** false
- **Human Verified:** false
- **Integration Status:** BLOCKED (Pending true evaluation)

## 4. Evaluation Metrics
### Validation Metrics (mAP50 / F1)
- **Phone Detector:** 94.48% / 89.34%
- **Seatbelt Detector:** 94.73% / 90.13%
- **Seatbelt Classifier:** Top-1 80.97%

### Frozen Test Metrics (mAP50 / F1)
- **STATUS:** INVALID_UNVERIFIED (Pending raw evaluation evidence. Test split data has not yet been reliably consumed.)

*Note: Detector mAP does not equal event accuracy.*

## 5. Behavior Evaluation
EVENT_EVALUATION_PENDING_GOVERNED_GROUND_TRUTH
Event accuracy is not yet proven due to insufficient human-reviewed event GT.

## 6. Known Failure Modes & Limitations
- **Phone:** Hand near face/ear triggers false positives. Low light reduces recall.
- **Seatbelt:** Dark clothing blends with dark belts causing false negatives.
- **Classifier:** High ambiguity leads to `uncertain_or_occluded`. This class MUST be kept fail-closed (UNKNOWN != UNFASTENED).

## 7. Camera Generalization
CROSS_CAMERA_GENERALIZATION_NOT_PROVEN. This baseline has not been proven against external camera domains.

## 8. Intended Use and Prohibited Interpretation
- **Intended Use:** Review candidate for Driver Monitoring Systems.
- **Prohibited Interpretation:** Do not map `unknown` to `unfastened`. Do not assign violation to passengers or outside pedestrians. Do not use test metrics to choose models.

## 9. Artifact Hashes
- **Phone Detector SHA256:** `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`
- **Seatbelt Detector SHA256:** `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500`
- **Seatbelt Classifier SHA256:** `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4`
- **Calibration Lock SHA256:** `40e6e7205fcbd2344f6d9d96b2fde7fe2b5582f15958981019772b7ca19b6a08`

## 10. Training Environment
NVIDIA GeForce RTX 3090, 24GB VRAM, Windows 10, PyTorch 2.8.0+cu128.
