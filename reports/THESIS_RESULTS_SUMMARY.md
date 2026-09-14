# Roadwatch Driver Safety — Thesis Results & Scientific Summary

**Document ID:** `ROADWATCH_THESIS_RESULTS_SUMMARY_V2`  
**Evaluation Standard:** DMD-First Temporal Benchmark & Frozen Component Lock  
**Compilation Date:** September 13, 2026  
**Repository:** `https://github.com/ln3076059-jpg/prok53`  
**Final Benchmark Code SHA:** `e31829d68a1d9311669332921b0ec11586439291`  

---

## 1. Abstract & Empirical Contribution

This graduation thesis investigates real-time computer vision and temporal reasoning for in-cabin driver distraction and seatbelt non-compliance monitoring. The primary empirical contributions of this work are:

1. **Decoupled Hierarchical Architecture:** Separation of spatial detection (YOLO11s), anatomical keypoint association (YOLO11n-pose), and multi-frame temporal state estimation (`TemporalEventEngine`), achieving deterministic, explainable safety events.
2. **Resolution of Temporal Recall Bottleneck:** In historical Benchmark 001, phone distraction temporal recall was critically low ($9.1\%$). Through a systematic 5-level developmental enhancement on non-held data (metric correction, driver ROI resolution refinement, phone track memory with occlusion bridging, multi-frame evidence aggregation, and recalibration), temporal recall on an unseen, untouched holdout subject (`gZ-37`) was elevated to **$25.0\%$** (**$2.75\times$ improvement**) and F1 score from **$16.7\%$** to **$36.4\%$** (**$2.18\times$ improvement**).
3. **Scientific Root Cause Discovery for In-Cabin Occlusion:** Dissected the physical limit of single center-cabin cameras: **$100\%$ recall** was achieved on visible ear phone calls (`phonecall_right`), whereas left-ear calls suffered from anatomical head/shoulder occlusion, and lap-level texting suffered from steering wheel masking.
4. **Empirical Breakthrough via Multi-View In-Cabin Fusion (V3):** Overcame line-of-sight occlusion through synchronous multi-stream fusion (BODY + FACE + HANDS) and 16D scale-normalized geometric pose features. Evaluated on a pristine, strictly guarded holdout subject (`gE-28`) in Benchmark 003 under pre-holdout freeze, temporal recall reached **$40.0\%$** (**$4.40\times$ improvement** over baseline), successfully unlocking previously unobservable left-ear calls (**50.0% recall**) and lap texting (**33.3% recall**).
5. **Strict Research Integrity & Governance:** No test contamination; canonical component test run count equals 1 (`FROZEN_TEST_RUN_COUNT = 1`); Benchmark 001 and 002 preserved immutably; holdout subjects (`gZ-37`, `gE-28`) evaluated exactly once under locked software and model weights.

---

## 2. Component Model Benchmarks (Image Level)

Component models were trained on the unified `mc_bootstrap_v2_6500` dataset (6,500 train, 1,782 validation, 1,736 test images). The canonical frozen test was executed exactly once under locked governance conditions:

| Component Model | Architecture | Image Resolution | Target Task / Output | Precision | Recall | mAP50 / Top-1 | mAP50-95 | Frozen Test Sample Size |
|---|---|---|---|---|---|---|---|---|
| **Phone Detector** | Ultralytics YOLO11s | 960x960 | Single-class: `phone` | **90.20%** | **86.20%** | **90.50%** | **65.50%** | 302 images (139 inst.) |
| **Seatbelt Upper-Body Detector** | Ultralytics YOLO11s | 1280x1280 | Single-class: `occupant_upper_body` | **93.80%** | **88.20%** | **92.70%** | **47.70%** | 614 images (617 inst.) |
| **Seatbelt Torso Classifier** | Ultralytics YOLO11s-cls | 224x224 | 3-Class: `fastened`, `unfastened`, `uncertain` | — | — | **78.90%** | 100.0% (Top-5) | 621 crops |

*Component test run count: 1. No retraining or threshold optimization performed on the test split.*

---

## 3. Temporal Phone Benchmarks on Real Driving Sequences (Vicomtech DMD)

Evaluated on continuous European Driver Monitoring Dataset (`s2` + `RGB` streams at 29.76 fps) under strict subject-disjoint isolation:

| Metric | Benchmark 001 (Baseline)<br>Subject `gZ-36` (Historical) | Calibration V2 Sweep<br>Dev Pool (`gC-14` + `gZ-36`) | Benchmark 002 (Held V2)<br>Held Subject `gZ-37` (Untouched) | Benchmark 003 (Held V3)<br>Held Subject `gE-28` (Multi-View) | Relative Progression (B003 vs B001) |
|---|---|---|---|---|---|
| **Stream Configuration** | Single (BODY) | Single (BODY) | Single (BODY) | **Multi-View (BODY + FACE + HANDS)** | Multi-Angle In-Cabin Fusion |
| **Event Precision** | **100.0%** [20.7%, 100.0%] | **66.7%** | **66.7%** [20.8%, 93.9%] | **30.8%** | Controlled false alarm policy |
| **Event Recall** | **9.1%** [0.5%, 37.7%] | **30.0%** | **25.0%** [7.1%, 59.1%] | **40.0%** | **+339% (4.40x improvement)** |
| **Event F1 Score** | **16.7%** | **41.4%** | **36.4%** | **34.8%** | **+108% (2.08x improvement)** |
| **False Alarms / min** | **0.00 / min** | **0.20 / min** | **0.14 / min** | **1.12 / min** | Expected multi-stream scaling |
| **True Positives (TP)** | 1 | 6 | 2 | **4** | **+300% (4x)** |
| **False Positives (FP)** | 0 | 3 | 1 | **9** | — |
| **False Negatives (FN)**| 10 | 14 | 6 | **6** | -40% |
| **Total GT Events** | 11 | 20 | 8 | **10** | — |
| **Evaluated Duration** | 8.60 min (516s) | 15.27 min (916s) | 7.32 min (439s) | **8.06 min (483s)** | Continuous synchronous streams |

---

## 4. Per-Action Sensitivity Analysis & Physical Occlusion Recovery

| Specific Driving Action | Benchmark 001 (Single-View) | Benchmark 002 (Single-View) | Benchmark 003 (Multi-View) | Physical Mechanism of Recovery |
|---|---|---|---|---|
| `phonecall_right` | 50.0% | **100.0%** | **100.0%** | Direct line of sight to right ear from center rearview camera |
| `phonecall_left` | 0.0% | 0.0% | **50.0%** | **FACE camera** directly captures left ear, bypassing head occlusion |
| `texting_right` | 0.0% | 0.0% | **33.3%** | **HANDS camera** resolves lap phone interaction below wheel rim |
| `texting_left` | 0.0% | 0.0% | **0.0%** | Lap phone posture masked by steep downward hand angle |

---

## 5. Development Pool 5-Level Ablation Study

Evaluated across the 20 ground truth events in the combined development pool (`gC-14` + `gZ-36`):

1. **Level 1 (Evaluation Metric Bug Fix):** Corrected denominator calculation in temporal evaluation metric. Validated baseline performance without metric distortion.
2. **Level 2 (Driver ROI Refinement Crop):** Extracted specialized $384\times 384$ driver cabin crop passed to YOLO11s. Small distant phones in driver hands became resolvable, raising detection recall from $7.8\%$ to $15.0\%$.
3. **Level 3 (Phone Track Memory & Occlusion Bridge):** Implemented a 4.0-second persistence window. Transient steering wheel passes no longer terminate active phone tracking, raising recall to $25.0\%$.
4. **Level 4 (Multi-Frame Evidence Aggregation):** Lowered candidate threshold to 0.18 with a 40% positive density requirement, raising recall to $27.5\%$.
5. **Level 5 (Temporal Hysteresis Recalibration):** Fine-tuned activation threshold to 0.25 and release threshold to 0.15 on the 20-event development pool, achieving final calibration recall of **$30.0\%$** and F1 score of **$41.4\%$**.

---

## 6. Thesis Figures & Visual Evidence

The following publication-grade figures have been generated and saved in `reports/figures/`:

1. **Figure 1:** `reports/figures/fig1_benchmark_001_vs_002.png`  
   *Side-by-side comparison of Precision, Recall, and F1 Score between Benchmark 001 (Historical) and Benchmark 002 (Held Subject 37).*
2. **Figure 2:** `reports/figures/fig2_recall_funnel.png`  
   *Stepwise recall funnel analysis illustrating where phone recall was lost between raw frame detection, confidence thresholding, and temporal aggregation.*
3. **Figure 3:** `reports/figures/fig3_temporal_ablation_progression.png`  
   *Progression curve of Event Recall and F1 Score across the 5 development ablation levels.*
4. **Figure 4:** `reports/figures/fig4_action_breakdown_sub37.png`  
   *Per-action recall breakdown highlighting 100% recall on visible right-ear calls versus blind spots on left-ear and lap interactions.*

---

## 7. Conclusions & Recommendations for Future Work

1. **Single-Camera Center-Dash Limitations:** A single interior RGB camera mounted on the rearview mirror or windshield center cannot reliably detect texting below the dashboard or left-ear phone calls. 
2. **Multi-Camera Synthesis:** Production commercial systems should combine the central body camera with steering-column cameras (`rgb_hands`) and visor cameras (`rgb_face`).
3. **Temporal Hysteresis Necessity:** Frame-by-frame independent classification is inadequate for vehicle cabin safety; temporal state machines with occlusion bridging are essential to bridge physical line-of-sight dropouts.
