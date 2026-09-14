# DMD V3.1 Development False Positive Root Cause Analysis

**Scope:** Development Pool Only (`gC-14`, `gZ-36`, `gB-9`)  
**Holdout Rule:** `gZ-37` and `gE-28` are permanently locked and excluded.  
**Purpose:** Map each development false positive to its originating module and component interaction signal trace.

---

## 1. Modular Attribution Matrix

Every false positive in the development experiments was systematically traced through the multi-stage pipeline:

```text
CAMERA STREAMS (BODY, FACE, HANDS)
       ↓
[STAGE 1: OBJECT DETECTORS]  ──────> Det FP (Weak crop proposals, e.g. wheel / chin patch)
       ↓
[STAGE 2: POSE ESTIMATION]   ──────> Pose FP (Hand near face confused with phone call)
       ↓
[STAGE 3: MULTI-VIEW FUSION] ──────> Fusion FP (Un-gated max view allows 1 weak view to trigger)
       ↓
[STAGE 4: TRACK MEMORY]      ──────> Track FP (4.0s bridge sustains ghost track past action end)
       ↓
[STAGE 5: TEMPORAL ENGINE]   ──────> Hysteresis / Split FP (3.0s gap split creates duplicate event)
```

### Module Contribution Table across 15 Development FPs:

| Primary Originating Module | Incident Count | % Total | Signal Signature | Key Contributing Signals |
| :--- | :---: | :---: | :--- | :--- |
| **`DUPLICATE_EVENT_SPLIT`** | 5 | 33.3% | True event split into 2 alerts | Extended duration (>30s), gap > 3.0s, IoU overlap on single GT |
| **`MULTIVIEW_FUSION`** | 4 | 26.7% | Auxiliary view trigger without BODY consensus | `eff_body = 0.0`, `eff_face > 0.25` or `eff_hands > 0.25`, no rescue gate |
| **`DETECTOR`** | 3 | 20.0% | Stray box on wheel hub / wristwatch / sunglasses | Low detector conf (0.18–0.24), transient presence (<1.0s) |
| **`POSE / ACTION_MODEL`** | 2 | 13.3% | Grooming / chin resting assigned call/text probability | `p_call = 0.40` default fallback; hand-to-ear proximity without phone |
| **`TRACK_MEMORY`** | 1 | 6.7% | Tail ghost after true call offset | Bridge TTL = 4.0s decayed past GT termination timestamp |

---

## 2. Signal Trace of Representative Failure Cases

### Case Study 1: `DUPLICATE_EVENT_SPLIT` (Temporal Fragmentation)
- **Subject:** `gZ-36`, continuous call interval `53.3s -> 110.7s` (duration: 57.4s).
- **Signal Trace:**
  - `t = 53.5s`: First event triggers (`fused_score = 0.82`, `eff_body = 0.88`, `pose_call = 0.91`). Matched as TP.
  - `t = 78.2s`: Driver rotates steering wheel with both hands temporarily for 3.4 seconds. `fused_score` drops below `release_threshold` (0.15). Active event closes at `t = 78.2s`.
  - `t = 82.1s`: Driver resumes phone call with right hand. `fused_score` rises to 0.79. New event candidate emitted at `t = 83.0s`.
  - `Result`: Under strict 1-to-1 matching, the second segment (`t = 83.0s -> 110.5s`) is classified as FP (`duplicate_preds`).
- **Resolution:** Cross-fragment merge threshold of 5.0 seconds connects these two segments into a single unified event.

### Case Study 2: `MULTIVIEW_FUSION` (Un-gated FACE View Spurious Trigger)
- **Subject:** `gB-9`, timestamp `t = 194.5s -> 198.2s` (neutral driving between phone calls).
- **Signal Trace:**
  - `BODY Detector`: `body_phone_conf = 0.0` (both driver hands visible on steering wheel).
  - `FACE Detector`: `face_phone_conf = 0.26` (visor edge / shadow near driver temple misclassified).
  - `Pose Vector`: `lw_le = 0.82`, `rw_re = 0.79` (hands at 10-and-2 position; NO ear proximity).
  - `V3 Fusion Head`: `max_view_phone_conf = max(0.0, 0.26 * 0.90) = 0.234`. Because `ph_face < 1.0`, action model assigned `p_call = 0.40`. Fused score: `0.55 * 0.234 + 0.25 * 0.40 = 0.229 + temporal = 0.27 >= 0.25`.
  - `Result`: Event generated despite primary BODY view showing hands resting safely on wheel.
- **Resolution:** Require BODY ear proximity (`lw_le < 0.35` or `rw_re < 0.35`) before FACE view can activate rescue mode.

### Case Study 3: `TRACK_MEMORY` (Ghost Track Trail)
- **Subject:** `gB-9`, true phone call ends at `t = 69.8s`.
- **Signal Trace:**
  - `t = 70.0s`: Driver lowers phone to cup holder. `body_phone_conf = 0.0`.
  - `Occlusion Bridge`: Because `occlusion_bridge_seconds = 4.0`, bridge maintained decaying confidence (`decay = 0.35 -> 0.22 -> 0.17`) until `t = 73.8s`.
  - `Result`: Event extended 4.0 seconds into neutral driving, causing start/end latency mismatch and duplicate tail detection.
- **Resolution:** Shorten occlusion bridge to 2.0s with steeper decay when hands leave ear zone.

---

## 3. Conclusions on Detector Retraining (Section 9)

- **Finding:** Only 3 out of 15 false positives (20%) originated from detector bounding box false alarms; 80% originated from downstream fusion permissiveness, track ghosting, and temporal fragmentation.
- **Governance Verdict:**
  `PHONE_DETECTOR_RETRAINING = NOT_REQUIRED`
  The YOLO11s detector component remains highly performant (90.2% precision on canonical test). Retraining is unnecessary; precision recovery must be executed in downstream fusion, rescue gating, and event deduplication.
