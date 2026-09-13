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
- **Dataset Role:** `TEMPORAL_DEVELOPMENT` (Canonical Eligible: False, Untouched Holdout Eligible: False)
- **Split Governance:** Strict subject-disjoint isolation (`SUBJECT_OVERLAP = 0`, `SHA_OVERLAP = 0`)

| Benchmark Phase | Target Subject(s) | Event Precision | Event Recall | Event F1 | False Alarms / min | Mean Onset Latency | Mean Offset Latency | Provenance Artifact |
|---|---|---|---|---|---|---|---|---|
| **Temporal Calibration Sweep** | `gC-14` | **100.0%** | **46.2%** | **63.2%** | **0.00 / min** | +18.02s | -21.19s | `reports/DMD_PHONE_TEMPORAL_CALIBRATION.json` |
| **Held-Subject Benchmark** | `gZ-36` | **100.0%** | **9.1%** | **16.7%** | **0.00 / min** | +10.38s | -40.46s | `reports/DMD_PHONE_TEMPORAL_BENCHMARK.json` |

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
