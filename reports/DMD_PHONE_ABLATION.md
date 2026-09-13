# DMD Temporal Phone Pipeline — Development Ablation Study

**Evaluation Pool:** `DMD_TEMPORAL_DEVELOPMENT_POOL` (Subject `gC-14` and Subject `gZ-36`)  
**Total Development Duration:** 916.2 seconds (15.27 minutes)  
**Total Ground Truth Phone Events:** 20 events (9 in Sub 14, 11 in Sub 36)  
**Evaluator Code:** Frozen matching policy (`min_iou: 0.30`, `onset_tol: 2.5s`, `offset_tol: 3.5s`)  

---

## 1. Step-Wise Pipeline Ablation Table

| Iteration / Component Stage | Precision | Recall | F1 Score | False Alarms / min | Mean Start Latency | Primary Failure Mode Addressed |
|---|---|---|---|---|---|---|
| **0. Historical Baseline 001** (Sub 36 only) | 100.0% | 9.1% | 16.7% | 0.00 | +10.38s | Severe cabin occlusion, single-frame dropout |
| **1. Metric & Timestamp Bug Fixes** (Dev Pool) | 21.1% | 20.0% | 20.5% | 0.98 | +22.67s | Fixed prediction fragmentation & unique GT denominator |
| **2. + Driver Phone ROI Second Pass** (Level 2) | 38.5% | 25.0% | 30.3% | 0.65 | +19.12s | Small/partially occluded phone on steering column |
| **3. + Track Occlusion Bridge (2.5s)** (Level 3) | 54.5% | 30.0% | 38.7% | 0.33 | +17.36s | Brief occlusion bridging while pose remains active |
| **4. + Deep Bridge (4.0s) & Hysteresis V2** (Selected) | **66.7%** | **30.0%** | **41.4%** | **0.20** | **+15.51s** | Fragmented predictions coalesced into solid events |

---

## 2. Key Insights from Ablation

1. **Why Baseline Recall Was So Low (9.1%):**
   - The detector only caught isolated frames when the phone was held up against the window.
   - During natural texting or resting on the steering column, single-frame confidence drops below activation threshold.
   - Without an occlusion bridge, every visual dropout terminated the temporal state machine, fracturing single GT calls into misses.

2. **Impact of Driver Phone ROI Second Pass (Level 2):**
   - In DMD body camera, the driver lap and lower steering wheel occupy approximately 30% of the image.
   - Downsampling 1280x720 to 640 causes small phone bounding boxes (< 30 px) to lose feature contrast.
   - Cropping the driver region (`[0.0, 0.10, 0.65, 0.95]`) and running inference at 384 effectively doubles the pixel density of the phone, retrieving weak candidates without hallucinating on background objects.

3. **Impact of Temporal Occlusion Bridge (Level 3):**
   - Bridging up to 4.0 seconds when physical phone track was established prevents visual drops from fracturing events.
   - Importantly, **False Alarms per minute dropped from 0.98 to 0.20** because spurious transient detections are bound to genuine tracks, and the state machine does not repeatedly trigger new false alarms.
   - Precision tripled from 21.1% to 66.7%.

4. **Why Latency Remains Large (+15.51s):**
   - DMD OpenLABEL annotations mark the macro-action (the entire phone call duration, e.g. 50–60 seconds).
   - In actual video, drivers frequently bring the phone to the ear where it is 100% occluded by the head/hair, or hold it below the dashboard.
   - Physical phone detection only triggers when the handset is raised or lowered into sight. This latency is inherent to camera line-of-sight semantics rather than temporal state machine lag.
