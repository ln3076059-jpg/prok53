# DMD V3 Pose Feature Ablation Report

**Phase:** Roadwatch V3 Multi-View Research  
**Component:** Upper-Body Driver Pose Geometry & Normalization  
**Model Base:** YOLO11n-Pose (`models/auxiliary/yolo11n-pose.pt`, SHA256: `fa3eb05c4889c2598380d19a2e3cfcf1a9528bb5eb4fbaf9b9e67d2ca85806c9`)  
**Status:** Complete / Validated  

---

## 1. Feature Representation & Spatial Invariance

Raw pixel distances in cabin cameras vary severely across participants due to seat track position, torso height, and camera mounting geometry. V3 implements **Shoulder-Width Normalized Geometry**:

$$\text{scale} = \max(\|\mathbf{p}_{\text{left\_shoulder}} - \mathbf{p}_{\text{right\_shoulder}}\|_2, \|\mathbf{p}_{\text{shoulder\_mid}} - \mathbf{p}_{\text{hip\_mid}}\|_2, 10.0)$$

All 16 upper-body spatial features are expressed relative to this anatomical scale:

| Feature Dimension | Anatomical Pair | Target Behavior Indicated |
| :--- | :--- | :--- |
| `left_wrist_to_left_ear` | Wrist(9) $\rightarrow$ Ear(3) | Left-hand calling posture ($< 0.55 \cdot \text{scale}$) |
| `right_wrist_to_right_ear`| Wrist(10) $\rightarrow$ Ear(4) | Right-hand calling posture ($< 0.55 \cdot \text{scale}$) |
| `left_elbow_angle_deg` | Shoulder(5) - Elbow(7) - Wrist(9) | Arm flexure angle ($< 85^\circ$ during phone call vs $\approx 150^\circ-180^\circ$ on wheel) |
| `right_elbow_angle_deg`| Shoulder(6) - Elbow(8) - Wrist(10)| Arm flexure angle |
| `phone_to_wrist` | Phone Center $\rightarrow$ Wrist(9, 10) | Confirms handheld phone grasp ($< 0.30 \cdot \text{scale}$) |
| `phone_to_face` | Phone Center $\rightarrow$ Face(0, 3, 4) | Confirms near-head phonecall ($< 0.45 \cdot \text{scale}$) |

---

## 2. Qualitative & Quantitative Development Observations

### A. Non-Distraction Suppression (Scratching & Grooming)
On development subjects `gC-14` and `gZ-36`, non-target intervals (`driver_actions/hair_and_makeup`, `driver_actions/reach_side`) frequently bring wrists near the face.
- **Detector-Only Baseline (A0):** Vulnerable to false triggers if non-phone items (wallets, badges) trigger low-confidence phone detector spikes near the head.
- **With Pose Features (A2):** The `OTHER_GESTURE` class recognizes hand-to-face proximity when physical phone bounding boxes are absent ($c < 0.15$), explicitly suppressing false alarm escalation.
- **Measured FA/min:** Maintained at $\le 0.93$/min on `gZ-36` and $0.0$/min on `gC-14`.

### B. The Physical Occlusion Ceiling on Center Camera
On `gZ-36`, ground-truth action breakdown showed:
- `phonecall_right`: **100.0% Recall** (clearly visible to center cabin camera).
- `phonecall_left`: **0.0% Recall** (occluded behind driver's head and left shoulder).
- `texting_right`: **0.0% Recall** (phone held down in lap behind steering wheel).
- `texting_left`: **0.0% Recall** (occluded by wheel rim and hand).

**Scientific Finding:** Pose features alone on a center-cabin camera **cannot** overcome physical optical occlusion. The left ear and the lower lap are mathematically outside line-of-sight from the center mirror mount. This establishes the absolute necessity of multi-view camera fusion (FACE view for left-ear calls, HANDS view for texting).
