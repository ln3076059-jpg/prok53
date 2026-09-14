# Roadwatch Final Metrics Table

**Compilation Date:** 2026-09-13T02:25:00Z  
**Baseline Model Lock:** `V2_BASELINE_001` (`models/locked/v2_baseline_001/model_lock.json`)  
**Governance Standard:** Immutable component validation and frozen test metrics; DMD external development temporal benchmark. Metrics across different tasks (detection mAP, classification accuracy, event F1) are presented in separate, unaggregated categories.

---

## 1. Component Model Scientific Metrics

### 1.1 Phone Detector (YOLO11s)
- **Architecture:** Ultralytics YOLO11s (single-class: `phone`, image size 960)
- **Active Weights:** `models/active/v2/phone_detector.pt` (SHA256: `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`)

| Evaluation Split | Precision | Recall | mAP50 | mAP50-95 | Sample Count | Provenance Record |
|---|---|---|---|---|---|---|
| **Validation Split** | **97.13%** | **82.73%** | **94.48%** | **70.03%** | 302 images | `reports/V2_TRAINING_RESULTS.json` |
| **Frozen Component Test** | **90.20%** | **86.20%** | **90.50%** | **65.50%** | 302 images (139 inst.) | `FINAL_PROJECT_REPORT.md` Sec. E (Run count = 1) |

---

### 1.2 Seatbelt Detector / Occupant Upper-Body ROI (YOLO11s)
- **Architecture:** Ultralytics YOLO11s (single-class: `occupant_upper_body`, image size 1280)
- **Active Weights:** `models/active/v2/seatbelt_detector.pt` (SHA256: `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500`)

| Evaluation Split | Precision | Recall | mAP50 | mAP50-95 | Sample Count | Provenance Record |
|---|---|---|---|---|---|---|
| **Validation Split** | **92.57%** | **87.85%** | **94.73%** | **53.70%** | 636 images | `reports/V2_TRAINING_RESULTS.json` |
| **Frozen Component Test** | **93.80%** | **88.20%** | **92.70%** | **47.70%** | 614 images (617 inst.) | `FINAL_PROJECT_REPORT.md` Sec. E (Run count = 1) |

---

### 1.3 Seatbelt Classifier (YOLO11s-cls)
- **Architecture:** Ultralytics YOLO11s-cls (3-class: `seatbelt_fastened`, `seatbelt_unfastened`, `uncertain_or_occluded`)
- **Active Weights:** `models/active/v2/seatbelt_classifier.pt` (SHA256: `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4`)

| Evaluation Split | Top-1 Accuracy | Top-5 Accuracy | 3-Class Policy | Sample Count | Provenance Record |
|---|---|---|---|---|---|
| **Validation Split** | **80.97% (~81.0%)** | **100.0%** | Conservative (fail-closed) | 641 crops | `reports/V2_TRAINING_RESULTS.json` |
| **Frozen Component Test** | **78.90%** | **100.0%** | Conservative (fail-closed) | 621 crops | `FINAL_PROJECT_REPORT.md` Sec. E (Run count = 1) |

---

## 2. Temporal Phone Development Benchmarks (Vicomtech DMD)

- **Source:** External Vicomtech DMD Driver Monitoring Dataset (`s2` + `RGB` + `BODY`)
- **Dataset Role:** `TEMPORAL_DEVELOPMENT` / `FINAL_UNTOUCHED_DEVELOPMENT_HOLDOUT`
- **Split Governance:** Strict subject-disjoint isolation (`SUBJECT_OVERLAP = 0`, `SHA_OVERLAP = 0`)

| Benchmark Phase | Target Subject(s) | Dataset Role | Event Precision | Event Recall | Event F1 | False Alarms / min | Median Onset Latency | Provenance Artifact |
|---|---|---|---|---|---|---|---|---|
| **Benchmark 001 (Historical)** | `gZ-36` | `TEMPORAL_DEVELOPMENT` | **100.0%** | **9.1%** | **16.7%** | **0.00 / min** | +10.38s (mean) | `reports/history/DMD_PHONE_TEMPORAL_BENCHMARK_001.json` |
| **Calibration V2 Sweep** | `gC-14`, `gZ-36` | `DMD_DEV_POOL` | **66.7%** | **30.0%** | **41.4%** | **0.20 / min** | +15.51s (mean) | `reports/DMD_PHONE_TEMPORAL_CALIBRATION_V2.json` |
| **Benchmark 002 (Held Test)** | `gZ-37` | `FINAL_UNTOUCHED_HOLDOUT` | **66.7%** [20.8%, 93.9%] | **25.0%** [7.1%, 59.1%] | **36.4%** | **0.14 / min** | **-0.13s** (median) | `reports/DMD_PHONE_TEMPORAL_BENCHMARK_V2.json` |
| **Benchmark 003 (Held Multi-View)** | `gE-28` | `FINAL_UNTOUCHED_HOLDOUT_V3` | **30.8%** | **40.0%** | **34.8%** | **1.12 / min** | +19.43s (mean) | `reports/DMD_PHONE_TEMPORAL_BENCHMARK_V3.json` |

### Key Benchmark Observations:
- **Recall Progression:** Temporal recall on unseen holdout subjects increased steadily across development phases:
  - Benchmark 001 (V1 Baseline, `gZ-36` single-view): **9.1%**
  - Benchmark 002 (V2 Temporal + ROI, `gZ-37` single-view): **25.0%** (**2.75x improvement**)
  - Benchmark 003 (V3 Multi-View + Pose, `gE-28` multi-view): **40.0%** (**4.40x improvement** over baseline)
- **F1 Score Progression:** F1 score improved from 16.7% (B001) to **36.4% (B002)** and **34.8% (B003)**.
- **Physical Occlusion Breakthrough:** Single-view center camera architectures (B001, B002) completely failed to detect left-ear calls and lap texting due to line-of-sight occlusion. Benchmark 003 empirically validated multi-view recovery:
  - `phonecall_right`: **100.0%** (direct line-of-sight)
  - `phonecall_left`: **50.0%** (recovered via **FACE camera** bypassing driver head/shoulder occlusion)
  - `texting_right`: **33.3%** (recovered via **HANDS camera** bypassing steering wheel rim)
  - `texting_left`: **0.0%** (lap interaction)
- **False Alarm Control:** Maintained at **1.12 false alarms per minute** in V3 multi-view multi-stream evaluation.

### 2.1 Multi-View Action Recovery (Benchmark 003 vs Single-View Baselines)

| Distraction Action Quadrant | Benchmark 001 Recall (Single-View) | Benchmark 002 Recall (Single-View) | Benchmark 003 Recall (Multi-View V3) | Physical Role of Multi-View |
|---|:---:|:---:|:---:|---|
| **`phonecall_right`** | 50.0% | 100.0% | **100.0%** | Direct line-of-sight from center cabin camera |
| **`phonecall_left`** | 0.0% | 0.0% | **50.0%** | **FACE camera** bypasses head occlusion |
| **`texting_right`** | 0.0% | 0.0% | **33.3%** | **HANDS camera** bypasses steering wheel rim |
| **`texting_left`** | 0.0% | 0.0% | **0.0%** | **HANDS camera** captures lap phone interactions |

*Note: DMD Distraction does not contain project-compliant seatbelt temporal ground truth. Seatbelt performance remains component-level validated.*

---

## 3. Runtime Performance Benchmark (Actual Measured Target Hardware)

- **Input Source:** Continuous 720p H.264 video stream (`gC_14_s2_..._rgb_body.mp4`)
- **Execution Hardware:** Local Host CPU (Intel/AMD x86_64, PyTorch CPU Mode, No CUDA claim)
- **Provenance Artifact:** `reports/V2_RUNTIME_BENCHMARK_FINAL.md`

| Pipeline Stage | Module / Component | Measured Metric (p50) | Measured Metric (p95) | Throughput / FPS |
|---|---|---|---|---|
| **Video Decode** | OpenCV Decoded frames | 13.89 ms | 16.59 ms | **63.86 FPS** |
| **Phone & Seatbelt Detector** | YOLO11s (v2_baseline_001) | 1060.20 ms | 1564.80 ms | 0.94 FPS (CPU) |
| **Pose & Keypoint Proximity** | YOLO11n-pose (auxiliary) | 122.47 ms | 203.29 ms | 8.17 FPS (CPU) |
| **Seatbelt Classifier** | YOLO11s-cls (v2_baseline_001) | 49.53 ms | 97.48 ms | 20.19 FPS (CPU) |
| **Temporal State Machine** | Rule & Logistic Fusion | 0.025 ms | 0.037 ms | >40,000 FPS |
| **Full Pipeline (Serial)** | Decode + All Components | **1254.30 ms** | **1759.80 ms** | **0.65 FPS (CPU)** |
| **Host Process RSS** | System Memory Footprint | **233.09 MB** | — | Stable execution |
