# DMD Phone Temporal Error Analysis

**Evaluation Set:** Held DMD Subjects (36)  
**Total Predictions:** 1 (1 TP, 0 FP)  
**Total True Events:** 11 (1 Detected, 10 Missed)  

---

## 1. False Positive Categorization (0 events)

| Error Category | Count | Primary Mechanism & Observed Failure Mode |
|---|---|---|
| **Hand Near Face / Steering** | 0 | Driver hand gesturing near ear or chin triggers high hand proximity; transient false detection in low lighting |
| **Mounted / Static Context** | 0 | Phone briefly visible on dashboard/mount without active interaction |
| **Cabin Lighting / Glare** | 0 | Window reflections simulating metallic phone edges |
| **Passenger Cross-Binding** | 0 | Prevented by driver-only ROI binding |

---

## 2. False Negative Categorization (10 events)

| Error Category | Count | Primary Mechanism & Observed Failure Mode |
|---|---|---|
| **Brief Quick-Glance Interaction** | 1 | Interaction duration $< 0.50$s filtered by temporal hysteresis window |
| **Severe Occlusion by Steering Wheel** | 9 | Phone held low behind steering column; detector confidence falls below activation threshold |

---

## 3. Mitigation & Recommendations for Future Field Validation
1. **Adaptive Hysteresis:** Dynamically lower activation threshold when hand-to-face proximity $> 0.85$ is sustained.
2. **Steering Wheel Keypoint Masking:** Explicitly track steering rim to discount lower-quadrant occlusions.
3. **Independent Real-Cabin Self-Capture:** Validate under Vietnamese urban lighting variations.
