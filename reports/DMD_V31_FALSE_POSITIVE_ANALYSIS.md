# DMD V3.1 Development False Positive Analysis

**Scope:** Development Pool Only (`gC-14`, `gZ-36`, `gB-9`)  
**Holdout Rule:** `gZ-37` and `gE-28` are strictly excluded from false positive analysis, mining, or parameter tuning.  
**Objective:** Dissect every false positive alert produced during V3 development experiments (A6/A7 configurations) to isolate root causes and guide V3.1 precision recovery.

---

## 1. Executive Summary & Category Breakdown

Across the V3 development runs on `gB-9` and `gZ-36` (A7 configuration), a total of **15 false positive events** were recorded (6 on `gB-9`, 9 on `gZ-36`):

| False Positive Category | Count | % of FPs | Source Subject(s) | Primary Stream(s) | Likely Originating Module | Recommended V3.1 Fix |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- |
| **`TEMPORAL_FRAGMENTATION`** | 5 | 33.3% | `gZ-36`, `gB-9` | BODY, FACE | Temporal Event Engine / Matching Policy | Cross-fragment gap merge & hysteresis extension |
| **`FACE_VIEW_FALSE_POSITIVE`** | 3 | 20.0% | `gZ-36`, `gB-9` | FACE close-up | Multi-View Fusion Head (Un-gated max) | Auxiliary Rescue Gating (require ear geometry) |
| **`HANDS_VIEW_FALSE_POSITIVE`** | 3 | 20.0% | `gZ-36`, `gB-9` | HANDS close-up | Multi-View Fusion Head (Un-gated max) | Auxiliary Rescue Gating (require lap interaction) |
| **`SCRATCHING_FACE / GROOMING`** | 2 | 13.3% | `gZ-36` | BODY, FACE | Pose Classifier / Action Fallback | Negative action filter; suppress when phone conf < 0.30 |
| **`MULTIVIEW_DUPLICATE`** | 1 | 6.7% | `gB-9` | FACE + HANDS | Event Engine (asynchronous onset) | Cross-view event identity deduplication |
| **`PHONE_TRACK_GHOST`** | 1 | 6.7% | `gB-9` | BODY | Track Memory (4.0s bridge too long) | Reduce track occlusion bridge TTL from 4.0s to 2.0s |

---

## 2. Detailed Category Analysis

### 2.1 Category 1: Temporal Fragmentation & Duplicate Segments (33.3%, 5 events)
- **Mechanism:** During extended true phone calls (e.g. 46s–58s in duration on `gZ-36` and `gB-9`), the driver temporarily shifts grip or turns their head. If the detection confidence momentarily dips below the `release_threshold` (0.15) for $> 3.0$ seconds, the active event terminates. When the signal recovers 1.5s later, a second event is generated.
- **Evaluation Impact:** Under 1-to-1 temporal matching policy, only the first segment is credited as True Positive (TP); the trailing segment is marked as an unmatched prediction (FP).
- **V3.1 Solution:** Implement cross-fragment merge for identical occupant/action when inter-event gap is $\le 5.0$ seconds, preventing duplicate splits of continuous behaviors.

### 2.2 Category 2: Un-gated Auxiliary FACE View Triggers (20.0%, 3 events)
- **Mechanism:** The close-up `FACE` stream focuses directly on the driver's head and visor. Natural facial textures, dark hair patches, sunglasses frames, or hand-to-jaw resting positions occasionally produce transient low-confidence phone detector proposals ($c \approx 0.18 - 0.28$).
- **V3 Flaw:** V3 computed `max_view_phone_conf = max(eff_body, eff_face * 0.90, eff_hands * 0.90)`. A stray FACE detection was treated with 90% parity to a direct BODY detection, immediately raising the fused evidence score without confirming head-hand proximity in the primary BODY perspective.
- **V3.1 Solution:** **Auxiliary Rescue Mode Gating**: The FACE camera may only elevate phone candidates if:
  1. Primary BODY camera indicates hand-to-ear proximity (`lw_le < 0.35` or `rw_re < 0.35`), OR
  2. FACE phone detection is exceptionally strong ($c \ge 0.45$) and supported by acute elbow angle.

### 2.3 Category 3: Un-gated Auxiliary HANDS View Triggers (20.0%, 3 events)
- **Mechanism:** The downward-facing `HANDS` camera captures the driver's hands, wrists, steering wheel spokes, and center console buttons. Metallic reflections, smartwatches, or dark wheel stitching occasionally produce weak phone candidate boxes ($c \approx 0.20 - 0.26$).
- **V3 Flaw:** The default action model fallback assigned `p_text = 0.40` whenever `ph_face >= 1.0`, artificially inflating `action_support` and passing the 0.25 activation threshold.
- **V3.1 Solution:** Require sustained multi-frame evidence in HANDS view ($\ge 3$ consecutive detections) and restrict HANDS rescue activation to cases where the primary BODY camera confirms hands are in the lap area rather than resting symmetrically at 10-and-2.

### 2.4 Category 4: Grooming & Non-Phone Facial Touch (13.3%, 2 events)
- **Mechanism:** Driver touches chin or scratches nose without a mobile phone present.
- **V3 Flaw:** Although `RuleGuidedPoseClassifier` included `OTHER_GESTURE`, weak background noise combined with hand proximity elevated `call_geom_score`.
- **V3.1 Solution:** Implement strict negative action suppression: if phone detector confidence is $< 0.30$, elevate `OTHER_GESTURE` and suppress `PHONECALL` probability to $< 0.10$.

### 2.5 Category 5: Phone Track Ghost Persistence (6.7%, 1 event)
- **Mechanism:** A 4.0-second occlusion bridge window (`occlusion_bridge_seconds: 4.0`) was retained from V2 single-view development. After a true phone call ended, the 4-second persistence window kept the phone state alive while the driver was already transitioning to neutral driving, extending the event beyond ground truth offset by $> 5$ seconds and generating a tail false positive.
- **V3.1 Solution:** Audit and reduce occlusion bridge duration to **2.0 seconds**, which is sufficient to bridge rapid head turns without sustaining ghost tracks.
