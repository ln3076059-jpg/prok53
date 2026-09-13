# Roadwatch Driver Safety — Final Academic Project Report

**Status:** **ACADEMIC & RESEARCH PROTOTYPE COMPLETE; ENGINEERING COMPLETE; PRODUCTION READY = FALSE**  
**Repository:** `https://github.com/ln3076059-jpg/prok53`  
**Base Commit:** `fc6cfa9fde7f4f3292749bd1564e9af7f81b6281`  
**Active Model Lock:** `V2_BASELINE_001` (`models/locked/v2_baseline_001/model_lock.json`)  
**Evaluation Standard:** DMD-First Temporal Development & Immutable Component Verification  

---

## 1. Project Objective

The Roadwatch Driver Safety project investigates multi-modal computer vision and temporal reasoning for automated driver distraction and occupant safety compliance. The engineering objective is to build a robust, end-to-end software and artificial intelligence pipeline capable of:
1. Detecting mobile phone usage by drivers while disambiguating static mounts, passenger activity, and non-phone hand-to-face gestures.
2. Assessing driver and passenger seatbelt compliance across a three-state semantic model (`FASTENED`, `UNFASTENED`, `UNCERTAIN_OR_OCCLUDED`) using torso/upper-body regions of interest.
3. Performing temporal fusion across video sequences to eliminate single-frame transient false positives.
4. Preserving cryptographically tamper-evident evidence packages (original keyframe, annotated keyframe, multi-second video clip, and metadata trace).
5. Enabling authenticated human-in-the-loop review and audit workflows via a modern web interface.

The project operates strictly within academic research governance boundaries: real-world production certification is explicitly disclaimed until independent, physically isolated field testing is conducted.

---

## 2. System Architecture

Roadwatch employs a decoupled, multi-stage hierarchical architecture to ensure modularity, explainability, and fail-closed uncertainty handling:

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

## 3. Dataset Strategy

To respect licensing boundaries and ensure empirical honesty, Roadwatch distinguishes four data categories:

1. **Static Component Datasets:**
   - Evaluated 25,851 candidate images across DMS, Mendeley Driver Risk, AnywayLabs synthetic DMS, and Roboflow Seatbelt v4.
   - Filtered into the `mc_bootstrap_v2_6500` split (6,500 train / 1,782 val / 1,736 test images) with zero cross-split SHA or perceptual hash leaks.
2. **Vicomtech Driver Monitoring Dataset (DMD):**
   - **Source:** External Vicomtech DMD (`EXTERNAL_DMD_VICOMTECH`).
   - **Role:** `TEMPORAL_DEVELOPMENT` (`CANONICAL_ELIGIBLE = false`, `UNTOUCHED_HOLDOUT_ELIGIBLE = false`).
   - **Profile:** Session `s2`, Channel `RGB`, Stream `BODY` (native resolution 1280x720).
   - **Actions Mapped:** `phonecall_right`, `phonecall_left`, `texting_right`, `texting_left` $\to$ `PHONE_USE`.
3. **Internal Reference Sequences (`v2_sequence_001`):**
   - Designated `TEMPORAL_DEVELOPMENT_ONLY` (Holdout Excluded). Used strictly for pipeline sanity checking.
4. **Future Self-Capture Sequences (`v2_sequence_002`):**
   - Designated `OPTIONAL_FUTURE_SELF_CAPTURE_VALIDATION`. Not required for academic project completion; zero synthetic capture data fabricated.

---

## 4. V1/V2 Model Components

The production-candidate model inventory is frozen in `models/locked/v2_baseline_001/` with exact SHA-256 cryptographic manifests:

| Component | Network Architecture | Input Size | Task / Classes | Active Weights Hash (SHA256) |
|---|---|---|---|---|
| **Phone Detector** | Ultralytics YOLO11s | 960x960 | Object detection: `phone` | `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54` |
| **Seatbelt Detector** | Ultralytics YOLO11s | 1280x1280 | Object detection: `occupant_upper_body` | `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500` |
| **Seatbelt Classifier** | Ultralytics YOLO11s-cls | 224x224 | Classification: `fastened`, `unfastened`, `uncertain_or_occluded` | `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4` |
| **Vehicle Detector** | YOLO11n (auxiliary) | 640x640 | Cabin contextual bounding | `3c8d35688abccae6f0bf76d0563b784a9596dd0240d9d6e8e5dfa2ff82d9218d` |
| **Pose Estimator** | YOLO11n-pose (auxiliary) | 640x640 | 17 keypoints for hand/face proximity | `fa3eb05c93ec6655c6543b59df3ba4184ec9c5fef91f8931b6eece6fcf4a30e8` |

---

## 5. Component Training

Training was executed on an NVIDIA GeForce RTX 3090 GPU (24 GB VRAM, Windows 10, PyTorch 2.8.0+cu128) under strict governance logging:
- **Phone Detector:** Trained for 50 epochs on `mc_bootstrap_v2_6500` phone subset. Stochastic gradient descent with CosineAnnealing learning rate schedule.
- **Seatbelt Detector:** Trained for 50 epochs on upper-body proposal annotations to localize the torso anchor box.
- **Seatbelt Classifier:** Trained for 40 epochs on cropped occupant torso images with random horizontal flip and color jitter.

---

## 6. Component Validation

Evaluation on the independent validation split yielded the following verified metrics (recorded in `reports/V2_TRAINING_RESULTS.json`):

- **Phone Detector (302 val images):**
  - Precision: **97.13%**
  - Recall: **82.73%**
  - mAP50: **94.48%**
  - mAP50-95: **70.03%**
- **Seatbelt Detector (636 val images):**
  - Precision: **92.57%**
  - Recall: **87.85%**
  - mAP50: **94.73%**
  - mAP50-95: **53.70%**
- **Seatbelt Classifier (641 val crops, 3-class):**
  - Top-1 Accuracy: **80.97% (~81.0%)**
  - Top-5 Accuracy: **100.0%**

---

## 7. Frozen Component Test

The canonical frozen test was executed exactly once under locked governance conditions (`FROZEN_TEST_RUN_COUNT = 1`). In accordance with scientific integrity rules, this benchmark is permanently locked and never rerun:

- **Phone Detector (302 test images, 139 instances):**
  - Precision: **90.20%**
  - Recall: **86.20%**
  - mAP50: **90.50%**
  - mAP50-95: **65.50%**
- **Seatbelt Detector (614 test images, 617 instances):**
  - Precision: **93.80%**
  - Recall: **88.20%**
  - mAP50: **92.70%**
  - mAP50-95: **47.70%**
- **Seatbelt Classifier (621 test crops):**
  - Top-1 Accuracy: **78.90%**
  - Top-5 Accuracy: **100.0%**

---

## 8. DMD Temporal Development

Vicomtech DMD provides real-cabin continuous multi-minute sequences for developing temporal event logic without self-capture video dependencies:
- Official Source Identifier: `EXTERNAL_DMD_VICOMTECH`
- Dataset Role: `TEMPORAL_DEVELOPMENT`
- Target Stream: `s2` + `RGB` + `BODY`
- Sequential timeline verification confirmed continuous presentation PTS, zero decode corruption, and full OpenLABEL timeline compliance.

---

## 9. Temporal Calibration

Development calibration sweeps were executed strictly across development calibration video (`gC-14`), freezing the evaluation set:
- **Parameter Grid Swept:**
  - Window lengths: 1.50s (`window_15f_fast`), 2.00s (`window_20f_balanced`), 2.50s (`window_30f_conservative`), 3.00s (`window_45f_strict`)
  - Activation thresholds: 0.32 to 0.40; Candidate thresholds: 0.28 to 0.35
  - Cooldown: 3.0s to 5.0s; Gap tolerance: 1.50s
- **Selected Configuration (`window_15f_fast`):**
  - `window_seconds = 1.50` (~45 video frames at 30 fps)
  - `min_positive_seconds = 0.50`
  - `min_observations = 2`
  - `positive_ratio = 0.50`
  - `candidate_threshold = 0.28`
  - `activation_threshold = 0.32`
  - `release_threshold = 0.20`
  - `cooldown_seconds = 3.0`
  - Selected Config SHA-256: `5d70dc287251f79cdf3ecf5019f007cf80d4a7a4162bb49e1bf22e496549199c`
  - Calibration Event Precision: **100.0%** (0 false positives)
  - Calibration Event Recall: **46.2%** (6 true positive events detected)
  - Calibration Event F1 Score: **63.2%**
  - False Alarms per Minute: **0.00 / min**

---

## 10. DMD Held-Subject Benchmark

The frozen temporal configuration was evaluated once against the held evaluation subject (`gZ-36`) without parameter retuning (`reports/DMD_PHONE_TEMPORAL_BENCHMARK.json`):
- **Evaluated Sequence Duration:** 8.60 minutes (516.0 seconds continuous video)
- **Held Subject Evaluated:** `36` (Male participant, low-contrast nighttime body stream)
- **Event Precision:** **100.0%** (1 TP, 0 FP, zero false alarms on safe driving)
- **Event Recall:** **9.1%** (1 / 11 true events detected; remaining missed due to lower-quadrant wheel occlusion)
- **Event F1 Score:** **16.7%**
- **False Alarms per Minute:** **0.00 / min** ($\le 1.00$ / min target achieved)
- **Mean Event Start Latency:** **+10.38 s**
- **Mean Event End Latency:** **-40.46 s**
- **Pose Availability Rate:** **100.0%**
- **Unknown Occupant Rate:** **0.0%** (Driver role correctly bound)
- **Governance Classification:** `DMD_EXTERNAL_DEVELOPMENT_BENCHMARK` (Not canonical production holdout).

---

## 11. End-to-End Runtime Test

The complete application pipeline was verified on continuous DMD body video via `tools.run_dmd_smoke_test`:
- Exercised: Video decode $\to$ cabin isolation $\to$ within-vehicle tracking $\to$ occupant role assignment $\to$ phone detection $\to$ pose estimation $\to$ temporal fusion $\to$ event emission $\to$ evidence generation $\to$ database persistence $\to$ human review simulation.
- Proven Artifact: `reports/DMD_END_TO_END_SMOKE_TEST.json` with status `PASS`.
- Full human review workflow confirmed: `Review(event_id, decision=CONFIRM, reviewer_type=HUMAN)`.

---

## 12. Backend and Frontend Implementation

- **Backend (FastAPI):**
  - Fully asynchronous REST API (`backend/api/routes.py`).
  - Strict JWT authentication, bcrypt password hashing, and role-based access control (`admin`, `reviewer`).
  - Analysis job creation, background processing queue, event filtering, evidence streaming, CSV export.
  - Fail-closed runtime validation preventing uncalibrated models from serving production requests.
- **Frontend (React / TypeScript):**
  - Enterprise dashboard with responsive dark mode and modern glassmorphism aesthetic.
  - Video upload, real-time job tracking, multi-event inspection, and side-by-side evidence preview.
  - Human review workflow supporting one-click `CONFIRM`, `REJECT`, and `NEEDS_REVIEW` actions.
  - Production bundle builds cleanly (`frontend/dist/` in 10.69s).

---

## 13. Evidence and Review Workflow

- **Evidence Packaging:**
  - Every candidate event generates a dedicated evidence bundle: `original_keyframe.jpg`, `annotated_keyframe.jpg`, `evidence.mp4` subclip, and `trace.json`.
  - Full-package SHA-256 hashing anchors each bundle to prevent tampering.
- **Human-in-the-Loop Governance:**
  - No model prediction is automatically deemed an indisputable legal violation.
  - All events enter `PENDING` review status; human reviewers append immutable review records.
  - Database enforces foreign key constraints and prevents history alteration.

---

## 14. Runtime Performance

Measured on genuine 720p continuous cabin video on local host hardware (`reports/V2_RUNTIME_BENCHMARK_FINAL.md`):

| Component | p50 Latency | p95 Latency | Throughput |
|---|---|---|---|
| Video Decode | 13.89 ms | 16.59 ms | **63.86 FPS** |
| Phone & Seatbelt Detector (YOLO11s) | 1060.20 ms | 1564.80 ms | 0.94 FPS (CPU) |
| Pose Estimator (YOLO11n-pose) | 122.47 ms | 203.29 ms | 8.17 FPS (CPU) |
| Seatbelt Classifier (YOLO11s-cls) | 49.53 ms | 97.48 ms | 20.19 FPS (CPU) |
| Temporal Fusion State Machine | 0.025 ms | 0.037 ms | >40,000 FPS |
| **Full Serial Pipeline (CPU Mode)** | **1254.30 ms** | **1759.80 ms** | **0.65 FPS** |
| **System Memory Footprint (RAM)** | **233.09 MB** | — | Stable |

---

## 15. Error Analysis

Comprehensive error categorization documented in `reports/FINAL_ERROR_ANALYSIS.md`:
1. **Mounted / Static Phones:** Mitigated via hand/face keypoint distance metrics in `classify_phone_context`.
2. **Hand-to-Ear Gestures Without Phone:** Suppressed by dual thresholding and temporal persistence gating.
3. **Passenger Phone Use:** Decoupled by geometric occupant association; passenger detections never trigger driver violations.
4. **Occluded Seatbelts (Fail-Closed):** Dark clothing, thick winter jackets, and backpack straps produce `UNCERTAIN_OR_OCCLUDED`, locked to `NEEDS_REVIEW`.

---

## 16. Security and Testing

- **Software Test Suite:** **454 tests passed / 0 failed** across unit, API, integration, and security layers.
- **Security Hardening:**
  - Path traversal protection on uploads and evidence retrieval.
  - CSV formula injection escaping (`_csv_safe`).
  - File MIME and size limit validations.
  - Constant-time password verification via native bcrypt.
- **Continuous Integration:** Automated `.github/workflows/ci.yml` validates linting, compilation, tests, and frontend build.

---

## 17. Limitations

1. **Self-Captured Canonical Video:** `v2_sequence_002` was not physically recorded; preserved as `OPTIONAL_FUTURE_SELF_CAPTURE_VALIDATION`.
2. **Seatbelt Ground Truth in DMD:** DMD Distraction RGB lacks verified seatbelt temporal labels; seatbelt validity is established at the component level.
3. **Hardware Acceleration:** Current local benchmarks reflect CPU execution (~0.65 FPS); edge automotive deployment requires TensorRT/CUDA acceleration.

---

## 18. Academic Completion Conclusion

ROADWATCH demonstrates an end-to-end driver-safety research system covering component detection/classification, occupant-aware phone reasoning, temporal event generation, evidence preservation, human-review workflow, and full-stack deployment.

Vicomtech DMD provides source-native continuous real-cabin video for temporal phone development and subject-disjoint development benchmarking. Seatbelt performance remains supported by component-level governed evaluation rather than DMD temporal labels.

The project is academically complete as an engineering and research prototype. It is not claimed as a production-certified monitoring system.
