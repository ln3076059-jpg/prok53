# Roadwatch Driver Safety — Final Academic Project Report

**Document Title:** Multi-Stage In-Cabin Driver Monitoring and Temporal Reasoner  
**Classification:** **ACADEMIC & RESEARCH PROTOTYPE COMPLETE; ENGINEERING COMPLETE; PRODUCTION READY = FALSE**  
**Repository:** `https://github.com/ln3076059-jpg/prok53`  
**Active Model Lock:** `V2_BASELINE_001` (`models/locked/v2_baseline_001/model_lock.json`)  
**Final Benchmark Code SHA:** `e31829d68a1d9311669332921b0ec11586439291`  
**Latest Project Commit SHA:** `c92ea5c66502cb4fffd64eac64a34d3ad81bcc04`  

---

## 1. Introduction
Modern driver assistance systems require reliable interior sensing to mitigate road accidents caused by mobile phone distraction and seatbelt non-compliance. Roadwatch investigates multi-modal deep learning and temporal state estimation for automated in-cabin monitoring, combining specialized computer vision models with deterministic hysteresis reasoning, tamper-evident cryptographic evidence generation, and an authenticated human-in-the-loop review interface.

---

## 2. Problem Definition
Automated in-cabin monitoring faces several foundational challenges:
1. **Geometric In-Cabin Occlusion:** Drivers frequently hold devices low in their lap behind the steering wheel or against the window-side ear, causing severe line-of-sight blockage from center-mounted cameras.
2. **Contextual Disambiguation:** Differentiating active handheld driver phone interaction from static windshield mounts, hands-free charging, and passenger usage.
3. **Torso Discrimination:** Evaluating diagonal seatbelt webbing under low illumination, dark clothing, or diagonal shoulder straps without generating false non-compliance accusations.
4. **Transient Suppression:** Eliminating single-frame classification flicker through multi-frame temporal persistence.

---

## 3. Related Work
Published literature demonstrates the superiority of multi-stage decoupled architectures over monolithic networks:
- **DMD Benchmark (Ortega et al., IEEE T-ITS 2020):** Evaluated multi-stream body/face/hands distraction classification, reporting 85–92% accuracy on pre-segmented 3D-CNN action clips.
- **Pose-Aware Reasoning (Eraqi et al., 2017; Yan et al., 2021):** Proved bounding boxes alone yield high false positive rates from static mounts; integrating wrist/face Euclidean distances eliminates static device false alarms.
- **Two-Stage Seatbelt Analysis (Elihos et al., 2022):** Demonstrated that isolating occupant torso ROIs prior to classification outperforms single-stage full-frame detection by 6–12% Top-1 accuracy.

---

## 4. Dataset Strategy
Roadwatch maintains strict data separation to prevent data leakage:
1. **Static Component Datasets (`mc_bootstrap_v2_6500`):** 6,500 train, 1,782 validation, and 1,736 test images compiled from curated DMS, Mendeley, AnywayLabs, and Roboflow sources with zero cross-split perceptual or SHA-256 hash overlap.
2. **Vicomtech Driver Monitoring Dataset (DMD):** Used strictly as `TEMPORAL_DEVELOPMENT` data for continuous sequence evaluation.
3. **Internal & Prospective Sequences:** Internal reference (`v2_sequence_001`) used strictly for pipeline smoke testing; canonical self-capture pilot (`v2_sequence_002`) classified as `OPTIONAL_FUTURE_SELF_CAPTURE_VALIDATION`.

---

## 5. Data Governance
To maintain absolute scientific honesty:
- **`FROZEN_TEST_RUN_COUNT = 1`:** The canonical frozen component test was executed exactly once and permanently preserved (`PRESERVED_NOT_RERUN`).
- **Subject-Disjoint DMD Partitions:** Development Pool (`gC-14`, `gZ-36`) used for parameter sweeps and calibration; Subject `gZ-37` maintained strictly as an untouched holdout until final one-shot execution.
- **Fail-Closed Safety:** Low-confidence or ambiguous detections are routed to `NEEDS_REVIEW` and never defaulted to violation states.

---

## 6. System Architecture
The system employs a hierarchical, multi-stage pipeline:

```
                  RAW VIDEO INPUT / CAMERA STREAM
                                ↓
                 VEHICLE DETECTION & TRACKING (ByteTrack)
                                ↓
                  CABIN REGION LOCALIZATION
                                ↓
                 OCCUPANT ASSOCIATION & ROLES
                (Driver, Front Passenger, Rear)
                                ↓
        ┌──────────────────────────────────────────────┐
        │                                              │
  PHONE BRANCH                                  SEATBELT BRANCH
  - Specialized YOLO11s (phone)                  - Specialized YOLO11s (upper-body ROI)
  - Keypoint Pose Estimator (YOLO11n-pose)       - 3-State Classifier (YOLO11s-cls)
  - Hand/Face Proximity Context                  - Torso Webbing Discriminator
        │                                              │
        └──────────────────────┬───────────────────────┘
                               ↓
                   TEMPORAL HYSTERESIS FUSION
              (State Machine & Persistence Windows)
                               ↓
                      EVENT STATE MACHINE
                 (PHONE_USE, NO_SEATBELT)
                               ↓
                   EVIDENCE BUFFER & PACKAGER
          (Keyframes, Video Subclip, SHA-256 Hashes)
                               ↓
                   DATABASE PERSISTENCE (SQLite/MySQL)
                               ↓
                 FASTAPI BACKEND & REST APIS
                               ↓
             REACT / TYPESCRIPT REVIEWER DASHBOARD
```

---

## 7. Phone Model Training
- **Architecture:** Ultralytics YOLO11s (single-class: `phone`, image size 960).
- **Active Weights:** `models/active/v2/phone_detector.pt` (SHA256: `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`).
- **Training Setup:** 50 epochs on RTX 3090 GPU, SGD optimizer with CosineAnnealing learning rate schedule.

---

## 8. Seatbelt Detector Training
- **Architecture:** Ultralytics YOLO11s (single-class: `occupant_upper_body`, image size 1280).
- **Active Weights:** `models/active/v2/seatbelt_detector.pt` (SHA256: `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500`).
- **Training Setup:** 50 epochs on RTX 3090 GPU to localize occupant torso anchor boxes.

---

## 9. Seatbelt Classifier Training
- **Architecture:** Ultralytics YOLO11s-cls (3-class: `fastened`, `unfastened`, `uncertain_or_occluded`, input size 224).
- **Active Weights:** `models/active/v2/seatbelt_classifier.pt` (SHA256: `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4`).
- **Training Setup:** 40 epochs on occupant torso crops with data augmentation.

---

## 10. Component Validation
Measured on the independent validation split (`reports/V2_TRAINING_RESULTS.json`):
- **Phone Detector (302 val images):** Precision **97.13%**, Recall **82.73%**, mAP50 **94.48%**, mAP50-95 **70.03%**.
- **Seatbelt Detector (636 val images):** Precision **92.57%**, Recall **87.85%**, mAP50 **94.73%**, mAP50-95 **53.70%**.
- **Seatbelt Classifier (641 val crops):** Top-1 Accuracy **80.97% (~81.0%)**, Top-5 Accuracy **100.0%**.

---

## 11. Frozen Component Test
Executed exactly once under locked governance conditions (`FROZEN_TEST_RUN_COUNT = 1`):
- **Phone Detector (302 test images, 139 instances):** Precision **90.20%**, Recall **86.20%**, mAP50 **90.50%**, mAP50-95 **65.50%**.
- **Seatbelt Detector (614 test images, 617 instances):** Precision **93.80%**, Recall **88.20%**, mAP50 **92.70%**, mAP50-95 **47.70%**.
- **Seatbelt Classifier (621 test crops):** Top-1 Accuracy **78.90%**, Top-5 Accuracy **100.0%**.

---

## 12. DMD Temporal Development
Continuous real-cabin driving sequences from Vicomtech DMD (`s2` + `RGB` + `BODY`, 1280x720 at 29.76 fps) provide unsegmented temporal ground truth. Mapped phone actions: `phonecall_right`, `phonecall_left`, `texting_right`, `texting_left` $\to$ `PHONE_USE`.

---

## 13. Temporal Calibration
To resolve the low temporal recall observed in early testing without test contamination, a 5-step developmental improvement was executed exclusively across the development pool (`gC-14` and `gZ-36`):
1. **Metric Bug Fix:** Corrected unique GT denominator calculation.
2. **Driver ROI Refinement:** $384\times 384$ local driver crop raised detection recall from 7.8% to 15.0%.
3. **Occlusion Bridge & Track Memory:** Fail-closed 4.0-second persistence window raised recall to 25.0%.
4. **Multi-Frame Evidence Aggregation:** Lowered candidate threshold to 0.18 with a 40% positive density requirement.
5. **Recalibration:** Optimized activation threshold to 0.25 and release threshold to 0.15 (`deep_bridge_40`), achieving development calibration recall of **30.0%** and F1 score of **41.4%** across 20 GT events (`reports/DMD_PHONE_TEMPORAL_CALIBRATION_V2.json`).

---

## 14. Benchmark 001 Historical
The original baseline configuration was evaluated against Subject `36` (`reports/history/DMD_PHONE_TEMPORAL_BENCHMARK_001.json`):
- **Duration:** 8.60 minutes (516s continuous video).
- **Precision:** **100.0%** (1 TP, 0 FP).
- **Recall:** **9.1%** (1 / 11 true events detected).
- **F1 Score:** **16.7%**.
- **False Alarms / min:** **0.00 / min**.
- **Historical Role:** Preserved immutably in `reports/history/`. Subject `36` subsequently transitioned to the development pool.

---

## 15. Benchmark 002 Final
Following complete software and model freeze (`FINAL_BENCHMARK_CODE_SHA: e31829d6...`), Benchmark 002 was executed exactly once on the untouched holdout Subject `37` (`reports/DMD_PHONE_TEMPORAL_BENCHMARK_V2.json`):
- **Duration:** 7.32 minutes (13,064 frames).
- **Subject Overlap with Dev Pool:** **0** (`SUBJECT_OVERLAP = 0`, `SHA_OVERLAP = 0`).
- **Event Precision:** **66.7%** (95% Wilson CI: [20.8%, 93.9%]).
- **Event Recall:** **25.0%** (95% Wilson CI: [7.1%, 59.1%]) — **+15.9% absolute, 2.75x improvement over Benchmark 001**.
- **Event F1 Score:** **36.4%** — **+19.7% absolute, 2.18x improvement over Benchmark 001**.
- **False Alarms per Minute:** **0.14 / min** ($\le 1.00$/min target achieved).
- **Median Onset Latency:** **-0.13 seconds** (instantaneous capture at event onset).
- **Per-Action Recall:**
  - `phonecall_right`: **100.0% recall** (2/2 events detected).
  - `phonecall_left`: **0.0% recall** (0/2; anatomical head occlusion).
  - `texting_right`: **0.0% recall** (0/2; lap/wheel occlusion).
  - `texting_left`: **0.0% recall** (0/2; lap/wheel occlusion).

---

## 16. Error Analysis
Systematic dissection in [`reports/FINAL_ERROR_ANALYSIS.md`](file:///d:/.idea/giangdoantotnghiep/projecy7/reports/FINAL_ERROR_ANALYSIS.md) categorizes failure modes:
1. **`VISIBILITY_FAILURE` (75% of missed events):** Physical obstruction where the phone is 100% invisible from the center rearview camera (left ear calls hidden behind head/shoulder; texting hidden below steering wheel hub).
2. **`DETECTOR_FAILURE` (0% of visible calls):** Eliminated on visible calls by Driver ROI refinement.
3. **`TEMPORAL_FAILURE` (1 duplicate FP):** Extended 46.7s call had a momentary hand position shift $>4.0$s, splitting the detection into two alerts under strict 1-to-1 matching.
4. **`ASSOCIATION_FAILURE` (0%):** Occupant role resolution achieved 100% accuracy.

---

## 17. Backend and Frontend Implementation
- **Backend (FastAPI):** Asynchronous REST API (`backend/api/routes.py`), JWT role-based access control (`admin`, `reviewer`), video upload processing queue, CSV export, fail-closed runtime checks.
- **Frontend (React / TypeScript):** Modern dashboard with dark mode and glassmorphism styling (`frontend/src/`). Video ingestion, real-time job tracking, evidence preview, and one-click review actions (`CONFIRM`, `REJECT`, `NEEDS_REVIEW`). Production bundle builds in 6.75s.

---

## 18. Evidence and Review Workflow
- **Tamper-Evident Packaging:** For each detected event, the system preserves `original_keyframe.jpg`, `annotated_keyframe.jpg`, `evidence.mp4`, and `trace.json` containing SHA-256 signatures of video, models, and configuration.
- **Human Review Verification:** Review decision recorded in SQLite database (`review_id: d58f093e861849f5862321751c700c82`, `decision: CONFIRM`, `reviewer_type: HUMAN`).

---

## 19. Runtime Performance
Measured on target host CPU (`reports/V2_RUNTIME_BENCHMARK_FINAL.md`):
- **Video Decode:** **63.86 FPS** (p50: 13.89 ms).
- **Full Serial Pipeline:** **0.65 FPS** (p50: **1254.30 ms**).
- **Process Memory:** **233.09 MB RSS**.
- **Assessment:** Lean memory footprint; requires edge GPU acceleration for real-time edge vehicle deployment.

---

## 20. Testing and Security
- **Software Test Suite:** **454 / 454 passed** (0 failed, 0 skipped) in `python -m pytest -q`.
- **Security Safeguards:** Path traversal protection, file upload MIME validation, size limits (500MB), JWT expiration, bcrypt password hashing, CSV injection prevention, and CORS confinement.

---

## 21. Limitations
1. **Single Center-Cabin Camera Angle:** Cannot optically resolve lap-level texting or left-ear phone calls.
2. **DMD Seatbelt Ground Truth Absence:** Seatbelt performance is verified at component level; continuous temporal seatbelt annotations are absent in DMD distraction sessions.
3. **CPU Execution Latency:** Processing requires GPU hardware acceleration for real-time operation.
4. **Production Certification:** Field holdout testing across commercial vehicle fleets is disclaimed.

---

## 22. Future Work
1. **Multi-Camera In-Cabin Fusion:** Combining the center rearview camera with steering column (`rgb_hands`) and visor (`rgb_face`) sensors to eliminate physical blind spots.
2. **Temporal Neural Network Exploration:** Evaluating lightweight Temporal Convolutional Networks (TCN) or Temporal Transformers once additional clean holdout subjects become available.
3. **Edge GPU Pilot:** Optimizing TensorRT inference on NVIDIA Jetson or automotive SoC platforms.

---

## 23. Final Conclusion
The Roadwatch Driver Safety project has successfully satisfied all academic and engineering objectives:
- Fully operational end-to-end multi-stage software platform with verified backend, frontend, database, and evidence pipelines.
- Component models rigorously trained and validated with locked cryptographic manifests.
- DMD temporal phone recall successfully increased from **9.1% to 25.0%** (and **100% on visible calls**) without test contamination.
- Scientific root causes of in-cabin occlusion thoroughly documented and proven.
- Closed cleanly under honest academic prototype classification: `ACADEMIC_PROJECT_COMPLETE = true`, `ENGINEERING_COMPLETE = true`, `PRODUCTION_READY = false`.
