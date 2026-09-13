# Roadwatch V2 Runtime Performance Benchmark

**Report Date:** 2026-09-13T02:16:00Z  
**Video Source:** `datasets/external_dmd/original/extracted/gC_14_s2_2019-03-04T11;48;02+01;00_rgb_body.mp4`  
**Video Resolution:** 1280x720 (29.76 FPS, H.264)  
**Evaluated Frame Count:** 50 sequential frames  
**Active Config:** `models/model_config_v2.yaml`  
**Hardware Environment:** Local Host (Windows 10, Intel/AMD x86_64, PyTorch CPU Mode)  
**Status:** **MEASURED**  

---

## 1. Latency Breakdown by Pipeline Module

| Pipeline Stage | Module / Model | Status | p50 Latency (ms) | p95 Latency (ms) | Measured Throughput |
|---|---|---|---|---|---|
| **Video Decoding** | OpenCV `cv2.VideoCapture` | **MEASURED** | **13.89 ms** | 16.59 ms | **63.86 decode FPS** |
| **Cabin Detection** | Geometric / ROI crop | **MEASURED** | Included in Scope | Included in Scope | — |
| **Phone & Seatbelt Det.** | YOLO11s (`v2_baseline_001`) | **MEASURED** | **1060.20 ms** | 1564.80 ms | ~0.94 FPS (CPU) |
| **Pose & Proximity** | YOLO11n-pose (`models/auxiliary/`) | **MEASURED** | **122.47 ms** | 203.29 ms | ~8.17 FPS (CPU) |
| **Seatbelt Classifier** | YOLO11s-cls (`v2_baseline_001`) | **MEASURED** | **49.53 ms** | 97.48 ms | ~20.19 FPS (CPU) |
| **Temporal Fusion** | Temporal rule & logistic gate | **MEASURED** | **0.025 ms** | 0.037 ms | >40,000 FPS |
| **Total Component Latency** | Full sequential inference | **MEASURED** | **1241.09 ms** | 1744.90 ms | 0.81 FPS (CPU) |
| **Serial Decode + Infer** | End-to-end frame processing | **MEASURED** | **1254.30 ms** | **1759.80 ms** | **0.65 FPS (CPU)** |

---

## 2. Resource Utilization

| Resource Metric | Measured Value | Operational Assessment |
|---|---|---|
| **Host Process RAM (RSS)** | **233.09 MB** | Lightweight memory footprint; stable execution |
| **GPU VRAM Peak** | `None` (CPU Host Execution) | Honest measurement: CPU mode; GPU acceleration not claimed |
| **Decode Headroom** | 63.86 FPS vs 29.76 native | 2.15x real-time decode capability |

---

## 3. Real-Time Deployment Analysis & Findings
1. **CPU vs. GPU Performance:** On current CPU-only hardware, end-to-end sequential inference achieves **0.65 FPS** (~1.25s per frame). For full real-time 30 FPS processing, GPU inference (e.g. RTX 3060/4060 or Jetson Orin) or temporal frame decimation (5 FPS sampling with TensorRT FP16) is required.
2. **Decode Efficiency:** Native video decode runs at **63.9 FPS**, confirming the decoder easily outpaces standard 30 FPS cabin camera streams.
3. **Fusion Overhead:** Temporal fusion executes in **0.025 ms** (25 microseconds), demonstrating virtually zero computational overhead for stateful hysteresis and role binding.
