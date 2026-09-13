# Detailed Occlusion and False Negative Root-Cause Analysis

**Analysis Timestamp:** 2026-09-13T12:39:50+07:00  
**Inspection Protocol:** Frame-level sampling at Start, 25%, 50%, 75%, and End of all 20 Ground Truth phone intervals across Development Pool (`gC-14` and `gZ-36`).  
**Total Inspected Sample Points:** 100 frames across 20 distinct GT events.

---

## 1. Visual Failure Mode Distribution

| Visual Root-Cause Classification | Sample Count | Percentage | Description & Observed Mechanism |
| :--- | :---: | :---: | :--- |
| **`PHONE_BEHIND_STEERING_WHEEL`** | 52 | **52.0%** | Device is located in the driver's lap or console area during retrieval/return or low texting. The steering wheel rim and steering column completely block line-of-sight from the BODY camera. |
| **`DETECTOR_LOW_CONFIDENCE`** | 24 | **24.0%** | Device is partially visible in the driver's palm or fingers, but low contrast against dark clothing/interior produces detector confidence below $0.25$ (typically $0.08 \le conf \le 0.22$). |
| **`PHONE_MOSTLY_OCCLUDED`** | 17 | **17.0%** | Driver is actively talking on the phone with device pressed to ear. Only fingertips and back cover edges are exposed; $80\%$ of phone body is hidden by head and hand. |
| **`PHONE_FULLY_VISIBLE`** | 4 | **4.0%** | Clear, unobstructed view of the device screen or body. Detector reliably triggers with $conf \ge 0.25$. |
| **`PHONE_PARTIALLY_VISIBLE`** | 3 | **3.0%** | Device is partially visible during transition from lap to ear. Detector fires with $0.15 \le conf < 0.25$. |

---

## 2. Event-by-Event Case Studies

### Case 1: Long Call (`phonecall_right`, Subject 36, 57.39 seconds)
- **Start (sec 53.29):** `PHONE_BEHIND_STEERING_WHEEL` — Driver reaches into jacket pocket behind steering wheel.
- **25% (sec 67.64):** `PHONE_FULLY_VISIBLE` — Driver lifts phone to ear, screen reflects cabin light ($conf = 0.65$). **TP event generated**.
- **50% (sec 81.99):** `PHONE_MOSTLY_OCCLUDED` — Phone pressed against ear; hand fingers wrap around bezel. Single-frame detector drops out.
- **75% (sec 96.34):** `PHONE_MOSTLY_OCCLUDED` — Continued speech position; camera sees only driver's wrist and ear.
- **End (sec 110.69):** `PHONE_BEHIND_STEERING_WHEEL` — Phone lowered back to console.

### Case 2: Brief Texting (`texting_right`, Subject 36, 0.81 seconds)
- Duration is 24 frames (~0.8s). Driver glances down at phone screen momentarily above lap.
- `TemporalEventEngine` required minimum $1.2$s duration in baseline, so brief $0.8$s interactions were filtered out by temporal duration gating.

### Case 3: Low Texting (`texting_left`, Subject 14, 8.47 seconds)
- Driver holds phone in left hand resting on thigh.
- The bottom edge of the steering wheel rim creates partial occlusion, suppressing raw detector confidence to $0.14$–$0.19$. In Benchmark 001, this was rejected by the $0.25$ candidate threshold.

---

## 3. Engineering Countermeasures Derived from Findings

1. **Lower Candidate Threshold with Pose Protection:** Lower candidate threshold to $0.15$ for observations that have verified driver pose and hand proximity.
2. **Driver ROI Second Pass:** Cropping the driver's seating region at $640\times 640$ effective resolution enhances small phone features on lap and hands, turning `DETECTOR_LOW_CONFIDENCE` ($0.12$) into solid detections ($0.35$).
3. **Temporal Occlusion Bridging:** When an active phone track is established and the driver's hand remains elevated near the ear, bridge up to $3.5$ seconds of hand micro-occlusion (`PHONE_MOSTLY_OCCLUDED`), maintaining track continuity through 50-second phone calls.
