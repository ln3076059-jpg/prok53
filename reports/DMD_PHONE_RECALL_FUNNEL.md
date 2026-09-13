# DMD Phone Recall Stage-Wise Funnel Analysis

**Analysis Timestamp:** 2026-09-13T12:39:30+07:00  
**Data Scope:** Development Pool Only (`gC-14` and `gZ-36`)  
**Total GT Positive Frames Evaluated:** 512 frames (at 1 FPS sample rate across 20 GT phone intervals)  
**Total Non-Target Frames Evaluated:** 398 frames  
**Benchmark Reference:** Explains the recall collapse in Benchmark 001 ($Recall = 9.1\%$).

---

## 1. Funnel Summary Table

| Pipeline Stage | Positive Observations | Stage Recall | Drop from Previous Stage | Primary Mechanism |
| :--- | :---: | :---: | :---: | :--- |
| **Ground Truth Eligible Frames** | 512 | 100.0% | — | Full temporal distraction intervals |
| **STAGE A: Raw Physical Phone Detector** | 129 | 25.20% | **-74.80%** | **Massive physical occlusion** (behind steering wheel rim / pressed to ear) |
| **STAGE B: Candidate Threshold ($conf \ge 0.25$)**| 40 | 7.81% | **-17.38%** | Partial hand occlusions yield low confidences ($0.08 \le conf < 0.25$) |
| **STAGE C: Driver / Cabin ROI Containment** | 40 | 7.81% | 0.00% | 100% of detected phones are inside driver cabin |
| **STAGE D: Occupant Association** | 40 | 7.81% | 0.00% | 100% of cabin phones bind to driver occupant |
| **STAGE E: Pose / Hand / Face Context Gate** | 40 | 7.81% | 0.00% | 100% of driver phone detections are in hand/face proximity |
| **STAGE F: Phone Context Score Fusion** | 40 | 7.81% | 0.00% | Rule fallback & fusion retain all verified driver phone observations |
| **STAGE G: Temporal Candidate Accumulation** | 18 | 3.52% | **-4.30%** | Discrete frame drops pull mean window score below candidate threshold |
| **STAGE H: Activated Temporal Event** | 14 | 2.73% | **-0.78%** | Strict activation threshold ($0.32$) and lack of occlusion memory bridge |

---

## 2. Key Insights: Where Recall Collapses

### 1. Stage A & B are the Primary Bottlenecks (-92.19% cumulative drop)
Out of 512 GT phone frames, only 40 frames ($7.81\%$) exceed the active detector threshold $conf \ge 0.25$. In 74.8% of frames, the phone is physically obscured by the steering wheel column or the driver's hand/ear, preventing single-frame object detectors from firing with high confidence.

### 2. Context & Association Stages are Flawless (0.00% drop)
Stages C, D, E, and F produced **zero false rejections**. When a physical phone is detected in the driver area, occupant association correctly assigns it to the driver, and pose estimation confirms hand/ear proximity. The system does not suffer from false occupant rejections or bogus context gating.

### 3. Stage G & H Suffer from Zero Temporal Memory (-5.08% drop)
In Benchmark 001, isolated positive detections separated by 1–2 occluded frames could not trigger an event because the temporal window required unbroken continuous detections. Adding a temporal occlusion bridge is mandatory to bridge these micro-occlusions.
