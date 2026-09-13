# Related Work and System Design Decisions

This document summarizes the scientific literature, external datasets, and comparative design decisions behind the Roadwatch Driver Safety system.

In compliance with project scientific integrity rules, external published results are strictly separated from Roadwatch implementations and actual measured metrics.

---

## 1. Literature Comparison Matrix

### 1.1 Vicomtech Driver Monitoring Dataset (DMD)
- **Reference:** Ortega et al., *"DMD: A Large-Scale Multi-Modal Driver Monitoring Dataset for Attention and Distraction Analysis"*, IEEE Transactions on Intelligent Transportation Systems, 2020.
- **PUBLISHED RESULT:**
  - 37 subjects driving real car in daylight and simulated night conditions.
  - Multi-stream: RGB, Depth, IR from body, face, and hands cameras.
  - Distraction activity classification benchmark reported accuracy ~85-92% on pre-segmented action intervals using 3D-CNNs (e.g. I3D, SlowFast).
- **ROADWATCH IMPLEMENTATION:**
  - Adapts official ASAM OpenLABEL / VCD annotation schema (`training/dmd/adapter.py`) without modifying raw annotations.
  - Focuses on continuous s2 RGB BODY sequences with phone actions (`texting_left`, `texting_right`, `phonecall_left`, `phonecall_right`) and non-target intervals (`safe_drive`, `reach_side`, `hair_and_makeup`).
  - Implements subject-disjoint temporal calibration and evaluation split (gC-14/gZ-37 for calibration; gZ-36 for held evaluation).
- **ROADWATCH MEASURED RESULT:**
  - Phone event precision: Evaluated via `training.dmd.evaluate_temporal`.
  - Component Phone Detector mAP50: 94.5% (Validation), 90.5% (Frozen Test).
  - Runtime decode on DMD 720p H.264 stream: 63.9 FPS.

---

### 1.2 Pose-Aware Phone Use Reasoning vs. Naive Detection
- **Reference:** Eraqi et al., *"Driver Distraction Identification with an Ensemble of Convolutional Neural Networks"*, IEEE Intelligent Vehicles Symposium, 2017; Yan et al., *"Driving distraction detection with smartphone sensor data and vision"*, 2021.
- **PUBLISHED RESULT:**
  - Pure bounding-box phone detection produces high false positive rates due to mounted navigation phones, wallet cards, or passengers holding devices.
  - Multi-modal fusion combining hand-to-head distance and gaze vectors improves distraction F1 by 8-15%.
- **ROADWATCH IMPLEMENTATION:**
  - Two-stage geometric context reasoning (`backend/ai/auxiliary.py:classify_phone_context`).
  - Uses YOLO11n-pose keypoints to compute normalized Euclidean distance from phone bounding box to left/right wrist keypoints and facial keypoints.
  - Semantic states: `NO_PHONE`, `PHONE_PRESENT_NOT_USED`, `MOUNTED_OR_STATIC_PHONE`, `PHONE_USE`, `UNKNOWN_PHONE_CONTEXT`.
  - Fail-closed gate: Passenger phone detections are filtered out; only driver-bound phone use with confirmed hand/face proximity triggers `PHONE` violations.
- **ROADWATCH MEASURED RESULT:**
  - Pose estimation latency on CPU: p50 122.5 ms.
  - Phone context classification execution: $<0.1$ ms.
  - Eliminates false positive event triggers from static mounts.

---

### 1.3 Two-Stage Seatbelt Analysis: Occupant ROI vs. Full-Frame Detection
- **Reference:** Elihos et al., *"Seat Belt Usage Detection Using Deep Learning"*, IEEE Access, 2022; Zhou et al., *"Driver Seatbelt Wearing Detection Based on Deep Convolutional Neural Networks"*, 2017.
- **PUBLISHED RESULT:**
  - Full-frame thin-strip seatbelt detection suffers from high false negative rates due to dark clothing, low-contrast seatbelt webbing, and diagonal bag straps.
  - Two-stage pipeline (Torso / Occupant ROI detection followed by high-resolution classification) outperforms single-stage by 6-12% top-1 accuracy.
- **ROADWATCH IMPLEMENTATION:**
  - Stage 1: YOLO11s Seatbelt ROI detector localizing occupant upper-body region (`models/active/v2/seatbelt_detector.pt`).
  - Stage 2: YOLO11s-cls Seatbelt 3-class classifier (`fastened`, `unfastened`, `uncertain_or_occluded`).
  - Three-state semantic gate: `uncertain_or_occluded` is NEVER converted to `unfastened` violation; routed to `NEEDS_REVIEW`.
  - DMD Distraction RGB does not provide project-compliant seatbelt ground truth; seatbelt scientific evaluation is held at component validation/frozen test.
- **ROADWATCH MEASURED RESULT:**
  - Seatbelt Detector Validation: Precision 92.6%, Recall 87.8%, mAP50 94.7%, mAP50-95 53.7%.
  - Seatbelt Classifier Validation: Top-1 Accuracy 81.0%.
  - Frozen Component Test: Detector mAP50 92.7%; Classifier Top-1 78.9%.
  - Seatbelt Classifier inference latency on CPU: p50 49.5 ms.

---

## 2. Architectural Design Decisions

| Decision Area | Alternative Considered | Selected Approach | Rationale & Governance Justification |
|---|---|---|---|
| **Single vs. Multi-Model** | Single monolithic 3-class detector (V1) | Multi-stage decoupled V2 architecture | Allows specialized image sizes (960 for phone, 1280 for seatbelt), separate class training, and fail-closed isolation |
| **Temporal State Machine** | Raw frame thresholding | Hysteresis sliding window (`TemporalEventEngine`) | Prevents single-frame flicker; enforces minimum positive duration, gap tolerance, and cooldown period |
| **Temporal Source** | Waiting for self-captured video (`v2_sequence_002`) | DMD-First external development benchmark | Vicomtech DMD provides verified continuous real-cabin video with ground-truth distraction timestamps, enabling academic closure without inventing data |
| **Production Certification** | Claiming system is production-ready | Explicit `PRODUCTION_READY = false` | Scientific honesty: True production readiness requires physically independent governed field holdouts, which remain future work |
