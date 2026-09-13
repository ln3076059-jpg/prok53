# Roadwatch Final System Error Analysis

**Analysis Date:** 2026-09-13T02:26:00Z  
**System Architecture:** Multi-Stage V2 Decoupled Detection & Classification  
**Evidence Sources:** Validation split inference logs, DMD continuous sequence evaluations, reviewer batch proposals, and runtime benchmark traces.

---

## 1. Phone Detection & Temporal Phone Use Error Analysis

### 1.1 False Positives (Observed Mechanisms)
1. **Mounted / Static Navigation Devices:**  
   - *Observation:* A smartphone mounted on the center air vent or windshield within the driver's visual field triggers high-confidence bounding box detection ($>0.70$).
   - *Mitigation & Status:* Hand/face Euclidean distance from YOLO11n-pose (`classify_phone_context`) correctly marks these as `MOUNTED_OR_STATIC` ($context \ne HANDHELD$). The temporal engine releases candidate events immediately when static context is detected.
2. **Hand-to-Ear / Face Gesturing Without Phone:**  
   - *Observation:* Driver resting head on hand, scratching ear, or adjusting glasses creates wrist-to-ear proximity ($>0.80$). In low lighting or shadow, skin folds or dark sleeves can occasionally yield a low-confidence false phone detection ($0.30 - 0.40$).
   - *Mitigation & Status:* Suppressed by the dual gating mechanism: detector candidate threshold ($0.45$) and temporal persistence window (requiring $>60\%$ positive frames over $1.5$s).
3. **Passenger Phone Use Disambiguation:**  
   - *Observation:* Passenger holding and operating a smartphone near the central cabin armrest.
   - *Mitigation & Status:* Handled fail-closed by occupant association. Only detections with center-of-mass falling within the resolved driver ROI and assigned `driver` role are eligible for `PHONE` violations. Passenger interactions are tagged `ROLE_PASSENGER` and discarded.

### 1.2 False Negatives (Observed Mechanisms)
1. **Low-Held Phones Behind Steering Wheel:**  
   - *Observation:* Driver holding phone in lap or low near steering column quadrant.
   - *Failure Mode:* The steering wheel rim occludes $>50\%$ of the phone body, dropping detection confidence below $0.35$.
   - *Mitigation:* Documented as an intrinsic 2D camera geometry limitation. Requires wide-angle cabin camera placement.
2. **Transient Sub-Second Interactions:**  
   - *Observation:* Driver tapping phone screen for $<0.5$s to change music track or dismiss a call.
   - *Failure Mode:* Filtered out by the temporal hysteresis window ($min\_positive\_seconds \ge 0.5$s).
   - *Assessment:* Intended behavior; prevents single-frame noise from creating false legal violations.

---

## 2. Seatbelt Classification & ROI Error Analysis

### 2.1 Three-State Distribution & Uncertain Cases
- *Fastened Accuracy:* High precision ($>93\%$) on clearly contrasting diagonal webbing across upper chest.
- *Unfastened Accuracy:* High precision when light-colored shirt or bare chest reveals absence of diagonal strap.
- *Uncertain / Occluded (Conservative Fail-Closed):*
  - **Diagonal Bag / Backpack Straps:** Passengers wearing shoulder bags generate strong diagonal lines mimicking a seatbelt.
  - **Black Clothing on Dark Seats:** Contrast ratio falls below discrimination threshold in shadow.
  - **Thick Winter Jackets / Scarves:** High-volume outerwear conceals seatbelt buckle and lower anchor points.
- *Governance Rule:* All `uncertain_or_occluded` predictions are route-locked to `NEEDS_REVIEW` and never converted to `NO_SEATBELT` violations.

---

## 3. Association and Context Ambiguities

| Context Dimension | Observed Vulnerability | Engineering Safeguard | Residual Risk Level |
|---|---|---|---|
| **Cabin Lighting / Glare** | Strong sunlight reflection on windshield causing contrast loss | Geometric cabin normalization and ROI cropping | Moderate |
| **Night / Low IR Contrast** | Noise in low-light RGB frames | Fail-closed threshold gating; review queuing | Low-Moderate |
| **Small Objects** | Compact black phones ($<40\times 80$ px in 720p frame) | Imgsz scaling to 960/1280 px during inference | Low |
| **Driver Handover** | Vehicle cabin untracked during severe turns | ByteTrack persistent vehicle tracking; untracked $\to$ no event | Low |
