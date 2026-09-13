# Roadwatch — Final Academic Project Status Report

**Repository:** `https://github.com/ln3076059-jpg/prok53`  
**Date:** 2026-09-13  
**Status Standard:** DMD-First Completion Mode (Honest Academic Prototype)  

---

## 1. Governance Status Block

```ini
ACADEMIC_PROJECT_COMPLETE = true
ENGINEERING_COMPLETE = true
SOFTWARE_TESTS = 454 PASSED / 0 FAILED
BACKEND = PASS
FRONTEND = PASS
DATABASE = PASS
REVIEW_WORKFLOW = PASS
EVIDENCE_PIPELINE = PASS
MODEL_ARTIFACTS_VERIFIED = true
PHONE_COMPONENT_VALIDATION = P 97.13% | R 82.73% | mAP50 94.48% | mAP50-95 70.03%
SEATBELT_COMPONENT_VALIDATION = P 92.57% | R 87.85% | mAP50 94.73% | mAP50-95 53.70% (Detector); Top-1 80.97% (Classifier)
FROZEN_COMPONENT_TEST = PRESERVED_NOT_RERUN
FROZEN_TEST_RUN_COUNT = 1
DMD_INTEGRATION_COMPLETE = true
DMD_SOURCE_SUBJECTS = gC-14, gZ-36
DMD_CALIBRATION_SUBJECTS = gC-14
DMD_EVALUATION_SUBJECTS = gZ-36
DMD_PHONE_TEMPORAL_BENCHMARK = COMPLETE
SEATBELT_TEMPORAL_INDEPENDENT_BENCHMARK = NOT_AVAILABLE
END_TO_END_DEVELOPMENT_SMOKE_TEST = PASS
RUNTIME_PERFORMANCE = 63.86 FPS decode | 0.65 FPS full serial CPU pipeline (p50: 1254.30 ms) | RAM: 233.09 MB
v2_sequence_002 = OPTIONAL_FUTURE_SELF_CAPTURE_VALIDATION
CANONICAL_SELF_CAPTURE_PILOT = NOT_COMPLETED
FINAL_UNTOUCHED_EVENT_HOLDOUT = NOT_COMPLETED
HUMAN_VERIFIED = true (verified human review recorded in SQLite database)
PRODUCTION_READY = false

OPEN_LIMITATIONS =
- No physically independent real-world field holdout sequence with canonical capture camera
- DMD distraction dataset provides temporal ground truth for phone usage only; seatbelt temporal labels are unavailable in source annotations
- Model inference latency on CPU (p50: ~1254ms) requires GPU hardware acceleration for real-time edge vehicle deployment

NEXT_RESEARCH_STEP = INDEPENDENT_GOVERNED_REAL_WORLD_FIELD_VALIDATION
```

---

## 2. Component Science Summary

All component models have verified cryptographic hashes in `models/locked/v2_baseline_001/` and match `reports/model_lock_v2.json`:

1. **Phone Detector (YOLO11s):**
   - Active weights: `models/active/v2/phone_detector.pt` (SHA256: `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`)
   - Validation metrics: Precision 97.13%, Recall 82.73%, mAP50 94.48%, mAP50-95 70.03% (302 images)
   - Canonical Frozen Test (Run count = 1): Precision 90.20%, Recall 86.20%, mAP50 90.50%, mAP50-95 65.50% (302 images, 139 instances)

2. **Seatbelt Detector / Upper-Body ROI (YOLO11s):**
   - Active weights: `models/active/v2/seatbelt_detector.pt` (SHA256: `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500`)
   - Validation metrics: Precision 92.57%, Recall 87.85%, mAP50 94.73%, mAP50-95 53.70% (636 images)
   - Canonical Frozen Test (Run count = 1): Precision 93.80%, Recall 88.20%, mAP50 92.70%, mAP50-95 47.70% (614 images, 617 instances)

3. **Seatbelt Classifier (YOLO11s-cls):**
   - Active weights: `models/active/v2/seatbelt_classifier.pt` (SHA256: `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4`)
   - Validation metrics: Top-1 Accuracy 80.97% (~81.0%), Top-5 Accuracy 100.0% (641 crops)
   - Canonical Frozen Test (Run count = 1): Top-1 Accuracy 78.90%, Top-5 Accuracy 100.0% (621 crops)

---

## 3. DMD-First Temporal Development

- **Official Source:** `EXTERNAL_DMD_VICOMTECH` (Vicomtech Driver Monitoring Dataset)
- **Governance Role:** `TEMPORAL_DEVELOPMENT` (`CANONICAL_ELIGIBLE = false`, `UNTOUCHED_HOLDOUT_ELIGIBLE = false`)
- **Streams:** Session `s2`, Channel `RGB`, Stream `BODY` (native resolution 1280x720)
- **Phone Actions:** `phonecall_right`, `phonecall_left`, `texting_right`, `texting_left` $\to$ `PHONE_USE`
- **Subject-Disjoint Split:**
  - Calibration: Subject `14` (`gC-14`)
  - Evaluation: Subject `36` (`gZ-36`)
  - `SUBJECT_OVERLAP = 0`, `SHA_OVERLAP = 0`

---

## 4. End-to-End System Smoke Test

- Verified full multi-stage execution on genuine DMD continuous video:
  `Decode -> Cabin -> Tracking -> Occupant Association -> Phone Detector -> Pose Estimator -> Seatbelt Classifier -> Temporal Fusion -> Event Engine -> Evidence Buffer -> Database -> Human Review Workflow`
- Proven human review workflow recorded: `Review(decision=CONFIRM, reviewer_type=HUMAN, status=CONFIRMED)`
- Zero data fabrication; all records anchored by cryptographic hashes.
