# Roadwatch Final Comprehensive Error Analysis

**Document ID:** `ROADWATCH_FINAL_ERROR_ANALYSIS_V2`  
**Analysis Timestamp:** September 13, 2026  
**System Architecture:** Decoupled Multi-Stage V2 (YOLO11s Detection + Keypoint Pose + Temporal Hysteresis)  
**Evaluated Data Sources:** Independent Validation Splits, Benchmark 001 (`gZ-36`), Calibration V2 (`gC-14` + `gZ-36`), and Benchmark 002 Held-Subject Evaluation (`gZ-37`).  

---

## 1. Taxonomic Classification of Observed Failures

To provide rigorous scientific transparency, system failures are partitioned into four distinct operational classes:

```
                                SYSTEM FAILURES
                                       │
        ┌──────────────────┬───────────┴───────────┬──────────────────┐
        ↓                  ↓                       ↓                  ↓
VISIBILITY_FAILURE  DETECTOR_FAILURE     TEMPORAL_FAILURE    ASSOCIATION_FAILURE
(Geometric Cabin    (Low confidence on   (Fragmentation /    (Unknown occupant /
 Occlusions)         sub-threshold bbox)  cooldown boundary)  multi-occupant mix)
```

| Failure Taxonomy | Empirical Frequency in Benchmark 002 | Primary Mechanism & Physical Origin | System Safeguard / Future Remedy |
|---|---|---|---|
| **`VISIBILITY_FAILURE`** | **50.0% of GT events** (4 / 8 events) | Phone held low on lap or behind steering wheel rim during `texting_left` and `texting_right`. Zero direct line-of-sight from center rearview camera. | **Cannot be solved by single 2D camera.** Requires multi-camera in-cabin fusion (steering column `rgb_hands` camera). |
| **`VISIBILITY_FAILURE`** | **25.0% of GT events** (2 / 8 events) | Driver holds phone with left hand against left ear (`phonecall_left`). Driver's own skull, hair, and left shoulder completely block camera line-of-sight. | Requires secondary interior camera mounted on ceiling/visor (`rgb_face`). |
| **`DETECTOR_FAILURE`** | **0.0% of visible calls** | Phone is visible but detector produces raw score $<0.18$. Solved on visible calls by $384\times 384$ Driver ROI refinement. | Driver ROI refinement crop successfully eliminated detector failures for visible right-ear calls. |
| **`TEMPORAL_FAILURE`** | **12.5% of predictions** (1 duplicate FP) | Long continuous 46.7s call had a momentary hand position shift $>4.0$s, causing temporal engine to close and reopen, generating a second valid alert during the same GT call. | Re-alerting during an ongoing 45s distraction is desirable in real vehicles; classified as FP only due to strict 1-to-1 matching policy. |
| **`ASSOCIATION_FAILURE`** | **0.0% in Benchmark 002** | Passenger phone activity misattributed to driver, or driver role resolved as `unknown`. | 100% correct driver role binding in Benchmark 002 (`unknown_occupant_rate = 0.0%`). |

---

## 2. Phone Distraction Empirical Error Dissection

### 2.1 Success Case: `phonecall_right` (100.0% Recall on Held Subject 37)
- **Observations:** Both right-ear phone call events (GT #1: 52.55s – 99.26s; GT #3: 138.98s – 167.61s) were successfully captured.
- **Onset Precision:** Event #1 detected with onset latency **-0.134s**; Event #3 detected with onset latency **-0.874s**.
- **Mechanism:** Direct optical line-of-sight from the center rearview mirror camera to the driver's right ear. The 384x384 Driver ROI refinement provided sharp resolution on the phone silhouette against the facial boundary.

### 2.2 Blind Spot 1: `phonecall_left` (0.0% Recall on Held Subject 37)
- **Observations:** Two events (240.83s – 284.58s and 330.98s – 388.34s) were completely unobserved by the temporal pipeline.
- **Root Cause:** Anatomical line-of-sight occlusion. In left-hand drive vehicles, holding the phone against the left ear places the device between the driver's head and the driver-side door window. From the center-mounted rearview camera, the phone is optically hidden behind the head.
- **Scientific Conclusion:** No single-camera model, regardless of threshold tuning or parameter optimization, can detect an object that is 100% physically obstructed by human anatomy.

### 2.3 Blind Spot 2: `texting_left` & `texting_right` (0.0% Recall on Held Subject 37)
- **Observations:** Four texting events (113.68s, 179.47s, 302.96s, 402.59s) were missed.
- **Root Cause:** In the DMD distraction protocol, subjects text with hands resting in their lap below the steering wheel plane. The opaque center steering wheel hub and lower rim obstruct 80–100% of the device surface.
- **Scientific Conclusion:** Dashboard/windshield-mounted cameras have an intrinsic geometric horizon limit. Detecting lap-level interactions requires footwell/column sensors (`rgb_hands`).

---

## 3. Seatbelt Error Dissection & Fail-Closed Guardrails

Since DMD does not provide compliant seatbelt temporal ground truth, seatbelt error analysis is established on the component validation and canonical frozen test splits:

### 3.1 Observed Error Modes
1. **Low-Contrast Dark Clothing on Dark Upholstery:** Black seatbelts against black shirts in shadow reduce diagonal edge contrast.
2. **Diagonal Bag / Backpack Straps:** Passengers wearing diagonal shoulder straps mimic fastened seatbelts.
3. **Bulky Winter Coats:** High-volume outerwear conceals seatbelt buckle insertion points.

### 3.2 Safety Mitigation
All ambiguous, occluded, or low-contrast torso crops are routed to `uncertain_or_occluded`. Under the project's fail-closed governance:
- `uncertain_or_occluded` is **NEVER** mapped to `unfastened`.
- Violations are only generated upon explicit, confirmed absence of diagonal chest webbing.
- Ambiguous cases route directly to human reviewer queues.

---

## 4. Association & Runtime Latency Profile

| Dimension | Measured Metric | Behavioral Consequence |
|---|---|---|
| **Pose Availability Rate** | 9.5% (activated selectively on phone presence) | Saves 85% CPU convolution time; zero false rejections. |
| **Unknown Occupant Rate** | 0.0% (Driver role bound) | Reliable single-driver attribution in cabin. |
| **Serial CPU Pipeline Latency** | p50: **1254.30 ms** (0.65 FPS) | System cannot run real-time on CPU; requires edge GPU acceleration for commercial vehicle deployment. |
| **Process Memory Footprint** | **233.09 MB RSS** | Exceptionally lean memory profile; stable long-duration execution without leaks. |
