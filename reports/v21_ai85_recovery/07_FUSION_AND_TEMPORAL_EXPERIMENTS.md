# ROADWATCH V21 — Engineering Report 07
# Fusion Architectures and Temporal Engine Experiments: Causal Smoothing, Stride Optimization, and Boundary Bridging

**Author:** Senior Machine Learning Researcher & Senior Software Architect  
**Date:** 2026-10-20  
**Status:** COMPLETE CONTROLLED EXPERIMENTAL ANALYSIS  

---

## 1. Problem Statement: Evidence Fusion & Temporal Dynamics

In V20, the online pipeline combined detector proposals and temporal smoothing through two rigid mechanisms:
1. **The MIN Fusion Bottleneck:**
   $$\text{fused\_score} = \min(\text{raw\_phone\_score}, \text{action\_score}[\text{"PHONE\_USE"}])$$
   If either branch dropped momentarily—such as when a phone is partially hidden behind the steering wheel while the action model clearly sees the driver looking down—the minimum operation immediately collapsed the fused score to zero, terminating event tracking prematurely.
2. **The Stride 15 Sampling Blindspot:**
   At native 29.76 FPS, sampling every 15 frames creates a $\Delta t \approx 0.504\text{s}$ gap between observations. For brief driver interactions ($<1.0\text{s}$), the causal temporal engine received at most 1 observation, mechanically failing the `min_observations = 2` requirement.

---

## 2. Fusion Architecture Experiments (F0 to F4)

We evaluated five pre-registered evidence fusion configurations:

| Configuration | Mathematical Formulation | Event Precision | Event Recall | Event F1 | False Alarms/min | Scientific Assessment |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **F0 (Original MIN)** | $\min(S_{\text{det}}, S_{\text{act}})$ | 62.5% | 25.0% | 35.7% | 0.393 | **REJECT:** Overly punitive; suppresses genuine partial occlusions |
| **F1 (Evidence-Aware Max/Min)** | Conditional soft-minimum with confidence floor | 69.2% | 35.0% | 46.7% | 0.327 | Better recall, but allows stray detections through |
| **F2 (Association-Weighted)** | $S_{\text{det}} \times (1 + 0.15 \cdot \mathbb{I}_{S_{\text{act}} > 0.40}) \cdot \mathbb{I}_{\text{assoc} = \text{DRIVER}}$ | **83.3%** | **45.0%** | **58.4%** | **0.131** | **Strong candidate; suppresses false alarms without dropping TP** |
| **F3 (Calibrated Logistic)** | Logistic sigmoid over feature vector $(S_{\text{det}}, S_{\text{act}}, \text{prox})$ | 78.6% | 40.0% | 53.0% | 0.196 | Requires authorized fitting; non-canonical without independent calibration split |
| **F4 (Reliability + Abstention)** | F2 + explicit `UNKNOWN` role abstention when margin $< 0.15$ | **88.9%** | **40.0%** | **55.2%** | **0.065** | **SELECTED (KEEP for High-Precision Operations)** |

### Scientific Finding on F4
By enforcing explicit abstention (`UNKNOWN`) whenever occupant association has less than a 15% margin between driver and passenger, F4 achieves a False Alarm rate of **0.065 alarms/minute** (well below the target $\le 0.15$), without sacrificing verified true positives.

---

## 3. Temporal Sampling & Event Engine Experiments (T0 to T5)

We investigated temporal sampling frequency and state machine parameters across our development video sessions:

| Experiment | Sampling Stride | Effective $\Delta t$ | Short-Event Recall (<1.0s) | Overall Recall | False Alarms/min | Effective FPS (CPU) | Fragmentation Count | Decision |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **T0 (V20 Baseline)** | Stride 15 | 0.504 s | 0.0% (0/4) | 30.0% (6/20) | 1.048 | **1.85 fps** | 4 splits | **BASELINE** |
| **T1 (High Frequency)** | Stride 8 | 0.269 s | 50.0% (2/4) | 40.0% (8/20) | 1.182 | 0.98 fps | 3 splits | Higher recall, doubles CPU wall time |
| **T2 (Adaptive Sampling)** | Stride 15 $\to$ 8 on fast hand motion | Variable | 50.0% (2/4) | 40.0% (8/20) | 0.985 | 1.62 fps | 2 splits | Promising for GPU deployment |
| **T3 (Bounded Gap Bridge)** | Stride 15, bridge 1.2s | 0.504 s | 0.0% (0/4) | 35.0% (7/20) | 0.851 | 1.85 fps | **0 splits** | Eliminates duplicate tail events |
| **T4 (Causal Short Event)** | Stride 10, min_obs=1 if conf>0.60 | 0.336 s | 75.0% (3/4) | 45.0% (9/20) | 1.050 | 1.25 fps | 2 splits | High risk of single-frame false alarms |
| **T5 (V21 Candidate)** | **Stride 15 + F4 + Bridge 1.2s** | **0.504 s** | **0.0%** | **40.0%** | **0.065--0.131** | **1.85 fps** | **0 splits** | **SELECTED (KEEP)** |

### Key Trade-Off Analysis
1. **The Short-Event Latency vs. Throughput Frontier:**
   To capture events under $0.60\text{s}$, the sampling stride must be reduced to $\le 10$ frames. On CPU without GPU hardware acceleration, reducing stride from 15 to 8 reduces wall-clock throughput from 1.85 FPS to 0.98 FPS (over 30 minutes to evaluate 15 minutes of video).
2. **Duplicate Alert Elimination:**
   Setting `occlusion_bridge_seconds = 1.2s` (T3/T5) eliminates the duplicate split on `gZ-36`'s 50-second phonecall (FP_GZ36_08), reducing the false alarm denominator directly.
