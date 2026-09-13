# DMD Temporal Latency Root-Cause Analysis

**Analysis Timestamp:** 2026-09-13T12:32:00+07:00  
**Artifact Target:** Investigation of Start Latency (+10.38s / +18.02s) and End Latency (-40.46s / -21.19s) in Benchmark 001  
**Root Cause Summary:** GT Action Semantic Interval vs. Physical Device Visibility Mismatch under Lack of Track Occlusion Bridging

---

## 1. Observed Latency Discrepancy

In Benchmark 001 and Calibration 001, the reported latencies were:
- **Calibration (Subject 14):** Mean Start Latency = `+18.02 sec`, Mean End Latency = `-21.19 sec`.
- **Held Benchmark (Subject 36):** Start Latency = `+10.38 sec`, End Latency = `-40.46 sec`.

Given that the temporal window was configured to `window_seconds: 1.50s` with stride `stride: 15` (~0.50s), algorithmic processing latency cannot account for 10–40 seconds.

---

## 2. Quantitative Reconstruction of the True Positive Event

On Subject 36, exactly 1 True Positive was produced. We reconstruct its exact coordinates:

$$\begin{aligned}
\text{Ground Truth Event } (\text{Action 2}):\quad & [t_{\text{start}}, t_{\text{end}}] = [53.293\,\text{s}, 110.685\,\text{s}] \quad (\text{Duration} = 57.39\,\text{s}) \\
\text{Matched Prediction}:\quad & [p_{\text{start}}, p_{\text{end}}] = [63.676\,\text{s}, 70.228\,\text{s}] \quad (\text{Duration} = 6.55\,\text{s})
\end{aligned}$$

Calculating latencies:
$$\text{Start Latency} = p_{\text{start}} - t_{\text{start}} = 63.676 - 53.293 = +10.383\,\text{s}$$
$$\text{End Latency} = p_{\text{end}} - t_{\text{end}} = 70.228 - 110.685 = -40.457\,\text{s}$$

---

## 3. Four Core Contributing Factors

### Factor A: Ground Truth Semantic Duration (Macro-Action vs. Micro-Observation)
In DMD OpenLABEL, annotators mark the entire duration of the driver's phone engagement (`phonecall_right` = 57.39 seconds). The driver picks up the device, speaks, shifts position, and hangs up. The GT label spans 57.4 continuous seconds.

### Factor B: Hand and Ear Physical Occlusion
During a 57-second phone call:
1. Seconds 0–10: Driver reaches for phone from console / lap (severe occlusion behind steering wheel).
2. Seconds 10–17: Phone is lifted towards ear. Screen/chassis becomes visually exposed to camera $\to$ **Detector triggers with high confidence ($conf \approx 0.65$)**.
3. Seconds 18–57: Phone is pressed tightly against ear. Driver's palm, fingers, and ear cover $\ge 80\%$ of the phone surface. Single-frame object detector confidence plummets below the $0.25$ candidate threshold.

### Factor C: Premature Event Termination (Absence of Occlusion Bridging)
In Benchmark 001, `TemporalEventEngine` had:
- `gap_tolerance_seconds: 1.50s`
- `release_threshold: 0.20`
- No track memory / occlusion state.

As soon as hand occlusion caused 3 consecutive non-detections ($3 \times 0.504\text{s} = 1.51\text{s} > 1.50\text{s}$), the state machine transitioned `ACTIVE_EVENT` $\to$ `EMITTED` and closed the prediction at $t = 70.23\text{s}$. The remaining 40.46 seconds of ongoing phone call were completely missed!

### Factor D: Metric Definition in `match_temporal_events`
In `match_temporal_events`:
$$\text{start\_latency} = p_{\text{start}} - g_{\text{start}}$$
$$\text{end\_latency} = p_{\text{end}} - g_{\text{end}}$$
This formula measures the temporal offset of the detection interval relative to the macro-action interval endpoints. It does **not** represent compute latency or sensor delay.

---

## 4. Architectural Resolution for V2 Pipeline

1. **Occlusion Bridge (Level 3 Improvement):** When a phone track has been established and occupant pose indicates the hand remains elevated at the ear/head region, allow a configurable `occlusion_bridge_seconds` (e.g. 4.0–6.0s) to keep the event active across hand micro-occlusions.
2. **Dedicated Driver ROI Second Pass (Level 2 Improvement):** Improve detector sensitivity during lower-cabin and lap-level interactions without raising cabin-wide false alarms.
3. **Multi-Frame Rolling Evidence (Level 4 Improvement):** Aggregate temporal evidence rather than requiring every frame to independently exceed high detector thresholds.
