# Roadwatch V3 / V3.1 — Runtime Truth Audit & Throughput Reconciliation

**Date:** 2026-09-14  
**Audit Purpose:** Reconcile reporting conflict between Benchmark 003 execution throughput (~0.9 FPS CPU) and synthetic microbenchmark claims (~18.6 FPS).  
**Governance Policy:** Zero tolerance for unsubstantiated "real-time capable" claims. Component microbenchmarks must never be presented as full end-to-end continuous video pipeline throughput.

---

## 1. Runtime Truth Table

| Evaluation Setting | Pipeline Scope | Execution Device | Input Streams & Resolution | Sampling Policy | Measured Latency (p50) | Effective Throughput | Classification & Evidence Status |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **Benchmark 003 Execution** | Full multi-view pipeline (Decode 3 streams + YOLO11s + Pose + Fusion + Event Engine) | **CPU** (x86_64, PyTorch CPU mode) | 3 synchronous streams (`BODY`, `FACE`, `HANDS`) @ 1280x720 | Sample stride = 15 frames (2.0 Hz video time) | **1060.84 ms / sampled frame** | **0.9 FPS** (Wall clock: 1022.5s for 483s video) | **FULL_E2E_MEASURED (VERIFIED)**<br>Source: `reports/DMD_PHONE_TEMPORAL_BENCHMARK_V3.json` |
| **V2 Baseline Runtime** | Serial single-view pipeline (Decode + ByteTrack + YOLO11s + Pose + Cls + State Machine) | **CPU** (x86_64, PyTorch CPU mode) | 1 stream (`BODY`) @ 1280x720 | Continuous full frame (stride = 1) | **1254.30 ms / frame** | **0.65 FPS** | **FULL_E2E_MEASURED (VERIFIED)**<br>Source: `reports/V2_RUNTIME_BENCHMARK_FINAL.md` |
| **V3 Multi-View Microbenchmark** | Isolated component stage forward passes without 3-stream PyAV demux / decode | **Estimated GPU / Microbenchmark** | 1 full frame (640p) + 2 ROI crops + 1 pose (480p) | Synthetic test batch | 51.2 ms (summed stages) | 18.6 FPS (theoretical sum) | **SYNTHETIC_STAGE_SUM_ESTIMATE (UNVERIFIED AS E2E)**<br>Source: `reports/DMD_V3_RUNTIME_BENCHMARK.md` |
| **V3 Staggered / Alternating View** | Alternating stream inference (1 aux view per frame) | **Estimated GPU / Microbenchmark** | 1 full frame + 1 ROI crop | Synthetic test batch | 30.5 ms (summed stages) | 30.8 FPS (theoretical sum) | **SYNTHETIC_STAGE_SUM_ESTIMATE (UNVERIFIED AS E2E)**<br>Source: `reports/DMD_V3_RUNTIME_BENCHMARK.md` |

---

## 2. Root Cause Analysis of Throughput Discrepancy

1. **Hardware Mode: CPU vs. GPU Acceleration:**
   - In actual benchmark runs on the local testbed (`py -3.11 training/dmd/v3_evaluator.py`), PyTorch executes on the host CPU in single-process mode. Ultralytics YOLO11s requires ~800–1000 ms per forward pass on CPU.
   - The 18.6 FPS figure reported in `reports/DMD_V3_RUNTIME_BENCHMARK.md` assumed CUDA/GPU acceleration (25.8 ms for YOLO11s, 18.2 ms for Pose). It did not reflect actual CPU execution.
2. **Video Decode & Tri-Stream Multiplexing Overhead:**
   - Decoding 3 concurrent 1280x720 H.264 video streams simultaneously (`BODY`, `FACE`, `HANDS`) via sequential OpenCV/PyAV demuxing incurs significant frame grab latency (~40–80 ms on CPU) that was omitted from stage-isolated microbenchmarks.
3. **Sample Stride vs. Continuous Throughput:**
   - Benchmark 003 ran with `sample_stride_frames: 15`. Processing every 15th frame took 1060.84 ms per inference, meaning the wall-time execution speed was 0.9 processed FPS (~2.1x slower than real-time playback).
   - Continuous 30 FPS processing of all frames on CPU would run at ~0.06x real-time (16x slower than real-time playback).

---

## 3. Scientific Integrity & Claim Corrections

- **Retraction of "Real-Time Capable on CPU":**
  The claim that the full synchronous multi-view V3 pipeline is "real-time capable" on host CPU is **RETRACTED**.
- **Correct Technical Formulation:**
  "On standard CPU hardware, the full multi-view pipeline achieves **0.9 FPS** under 15-frame sampling (~1060 ms/frame) and **0.65 FPS** in continuous un-sampled execution. Sustained real-time edge processing ($\ge 30$ FPS) strictly requires hardware acceleration (dedicated edge GPU, e.g., NVIDIA Jetson Orin / RTX embedded, or TensorRT INT8/FP16 quantization)."
- **Separation of Evidence in All Reports:**
  All project documentation now strictly distinguishes `FULL_E2E_MEASURED` benchmarks from `SYNTHETIC_STAGE_SUM_ESTIMATE` microbenchmarks.
