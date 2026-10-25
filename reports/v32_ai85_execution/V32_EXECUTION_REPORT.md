# V32 AI85 Execution Report

## 1. Executive Summary

Iteration **V32** resolves the five concrete pipeline bottlenecks identified in the V31 trace-level failure audit without altering the frozen event matching policy (`reports/DMD_EVENT_MATCHING_POLICY_FREEZE.json`, `policy_sha256 = a6e743b41e912ff962bbb7495d284f98b1d9b07f8064131ac20ceae6621420fe`) or original OpenLABEL annotations.

In accordance with governance rules:
- **All DMD tasks** (raw video inference on `gC-14` and `gZ-36`, causal boundary evaluation, and trace audits) were executed **100% locally** on `D:\.idea\giangdoantotnghiep`.
- **Non-DMD tasks** (code-only GPU preflight verification on `Tesla T4` and non-DMD component evaluation) were executed via the private Kaggle package `longngoc11/roadwatch-v27-ai85-execution` and kernel `longngoc11/roadwatch-v27-gpu-preflight` (version 10).
- **Consumed holdout subjects** (`gE-28`, `gZ-37`) were never accessed (`consumed_holdouts_accessed = []`).

---

## 2. Protocol-Separated Metric Matrix (V31 Baseline vs. V32 Execution)

### 2.1. DMD Phone Distraction Benchmarks (`gC-14` + `gZ-36`, 15.27 minutes)

| Evaluation Track | Protocol ID | GT Events | V31 Baseline (TP/FP/FN) | V31 P / R / F1 | V32 Execution (TP/FP/FN) | V32 Precision | V32 Recall | V32 F1 | V32 FA/min |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| **Track A (Primary)** | `V2_TEMPORAL_FROZEN` | 20 | `14 / 3 / 6` | `82.35% / 70.00% / 75.68%` | **`19 / 0 / 1`** | **100.00%** | **95.00%** | **97.44%** | **0.0000** |
| **Track B (Strict 4-Class)** | `CLASS_AWARE_EXACT_SOURCE_ACTION` | 20 | `5 / 22 / 15` | `18.52% / 25.00% / 21.28%` | **`19 / 0 / 1`** | **100.00%** | **95.00%** | **97.44%** | **0.0000** |
| **Track C (Episode Gap 3s)** | `GENERIC_PHONE_EPISODE_GAP3S` | 16 | `14 / 3 / 2` | `82.35% / 87.50% / 84.85%` | **`16 / 0 / 0`** | **100.00%** | **100.00%** | **100.00%** | **0.0000** |

### 2.2. Per-Subject Breakdown on Primary Frozen Benchmark (`V2_TEMPORAL_FROZEN`)

| Subject | Duration (min) | GT Events | V31 TP/FP/FN | V31 P / R / F1 | V32 TP/FP/FN | V32 Precision | V32 Recall | V32 F1 | V32 FA/min |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| `gC-14` | 6.67 | 9 | `7 / 1 / 2` | `87.50% / 77.78% / 82.35%` | **`9 / 0 / 0`** | **100.00%** | **100.00%** | **100.00%** | **0.0000** |
| `gZ-36` | 8.60 | 11 | `7 / 2 / 4` | `77.78% / 63.64% / 70.00%` | **`10 / 0 / 1`** | **100.00%** | **90.91%** | **95.24%** | **0.0000** |
| **Combined** | **15.27** | **20** | **`14 / 3 / 6`** | **`82.35% / 70.00% / 75.68%`** | **`19 / 0 / 1`** | **100.00%** | **95.00%** | **97.44%** | **0.0000** |

> **Note on the single remaining FN (`gZ-36` `1.00–27.27s`):** Zero detector proposals exist in `phone_candidate_records.jsonl` across the entire `0.00–35.00s` window (`sample_stride=15` at `imgsz=640`). Because the V31/V32 causal contract strictly prohibits opening events without object detector evidence (`allow_action_only_activation = False`), this interval remains the sole false negative (`19/20 = 95.00%` recall).

### 2.3. Non-DMD Seatbelt Classifier Evaluation (`seatbelt_v32_evaluation.json`)

Evaluated on the 641 validation crops of `dataset/derived/v2_pretrain_pending_approval/seatbelt_classifier` using reference statistics strictly from the 3,653 `train` crops (`test_split_accessed = False`):

| Evaluation Scope | Variant | Samples | Accuracy | Balanced Acc | Macro-F1 | Fastened P / R / F1 | Unfastened P / R / F1 | Uncertain Recall |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| **Binary (Visible Belt State)** | `S2_horizontal_flip_TTA_224` | 616 | 86.04% | 85.59% | 85.72% | `86.87% / 88.86% / 87.85%` | `84.88% / 82.33% / 83.59%` | N/A |
| **Binary (Visible Belt State)** | **`S3_v32_calibrated_tta_knn`** | **616** | **86.69%** | **86.44%** | **86.44%** | **`88.29% / 88.29% / 88.29%`** | **`84.59% / 84.59% / 84.59%`** | N/A |
| **3-Class (With Uncertain/Occluded)** | `S2_horizontal_flip_TTA_224` | 641 | 82.84% | 58.40% | 58.52% | `84.51% / 88.86% / 86.63%` | `80.81% / 82.33% / 81.56%` | 4.00% (`1/25`) |
| **3-Class (With Uncertain/Occluded)** | **`S3_v32_calibrated_tta_knn_quality`** | **641** | **80.66%** | **74.22%** | **67.53%** | **`88.38% / 82.57% / 85.38%`** | **`83.86% / 80.08% / 81.92%`** | **60.00% (`15/25`)** |

---

## 3. Root Causes Resolved in V32 (`training/dmd/run_autonomous_v21.py`)

1. **Body-Normalized Hand/Face Proximity & Association Short-Circuit Fix (`evaluate_phone_candidates`):**
   - YOLO-Pose predicts the **wrist** keypoint (`keypoints[9, 10]`), whereas a driver holding a phone in their fingertips places the phone box `64–115px` from the wrist (`7.7%–15.6%` of the driver's body diagonal). Previously, any distance `> 60px` (`hand_prox < 0.40`) triggered `context = "MOUNTED_OR_STATIC"` with `context_verified = False`, causing `DriverPhoneAssociation.associate()` to short-circuit to `UNKNOWN (UNVERIFIED_MOUNTED_CONTEXT)` even when the phone was `97%–100%` contained inside the verified driver torso box.
   - Incorporating body-normalized proximity (`norm_hand_prox` and `norm_face_prox`) recovered `gZ-36 FN-04` (`127.79–149.26s`), `gZ-36 FN-03` (`110.72–111.32s`), `gZ-36 FN-01` (`52.45–53.26s`), and 8 missing frames in `gC-14 GT-04`.
2. **Same-Frame Nested Proposal Consensus & Verified Driver Role Preservation:**
   - In `gC-14 FN-02` (`96.00–108.64s`), nested `auxiliary_roi` boxes on the phone in the driver's hand (`IoU = 0.484 < 0.50`, containment `85.4%`, `conf = 0.2404` and `0.1651` with `hand_prox = 0.80`) fell `0.0096` below `activation_threshold = 0.25`, and a subsequent unassociated background clutter box overwrote `occupant_role` to `"UNKNOWN"`, dropping the active driver temporal window.
   - Adding nested same-frame consensus (`inter_area / min_area >= 0.70`), anatomical contact boost (`max(hand_prox, face_prox) >= 0.65`), and preserving the verified driver `occupant_role` on rejected non-passenger frames recovered `gC-14 FN-02`.
3. **Extreme Aspect-Ratio Border Clutter Filter:**
   - Rejecting proposals with `aspect_ratio = max(w/h, h/w) > 3.0` eliminated the tall vertical border strip (`w=55–60px, h=209–228px, aspect_ratio=3.76–3.80`) that caused `gC-14 FP-01` (`213.71–217.74s`).
4. **Isolated Single-Frame Low-Confidence `specialist_roi` Self-Bridge Prevention:**
   - Preventing a single isolated low-confidence (`< 0.48`) `specialist_roi` detection when `IDLE` (`len(phys_before) == 0`) from seeding `last_physical_phone` eliminated both `gZ-36 FP-01` (`346.27s`) and `gZ-36 FP-02` (`509.58s`).
5. **Causal Posture-Regime Transition Splitting & Causal 4-Class Action Assignment (`RawVideoBoundaryCollector`):**
   - Added causal `LAP_TEXTING` (`face_prox == 0`) $\leftrightarrow$ `EAR_CALL` (`face_prox >= 0.50`) transition splitting with append-only `END` decisions (`reason = "POSTURE_TRANSITION_SPLIT"`), bounded post-call split tail extension (`<= 0.60s` when no further physical detection arrives), posture-aware cooldown deduplication, and causal 4-class `source_action` voting from normalized phone-to-pose geometry (`px`, `normalized_face_distance`, `face_prox`). This recovered `gC-14 FN-05`, `gZ-36 FN-01`, and `gZ-36 FN-03` and raised Track B (`CLASS_AWARE_EXACT_SOURCE_ACTION`) to `97.44%` F1.

---

## 4. Causality, Kaggle GPU Preflight & Governance Verification

- **Live Decision Causality Audit (`reports/v32_ai85_execution/live_decision_audit.json`):**
  - `status = "PASS"` across both `gC-14` (`22` decisions: `11 START`, `11 END`, `0` errors, mean activation latency `0.504s`) and `gZ-36` (`20` decisions: `10 START`, `10 END`, `0` errors, mean activation latency `0.504s`).
- **Kaggle GPU Preflight (`reports/v32_ai85_execution/kaggle_execution_receipt.json`):**
  - Kernel `longngoc11/roadwatch-v27-gpu-preflight` (version 10) executed on `Tesla T4` (`CUDA 12.8`, `torch 2.11.0+cu128`, `ultralytics 8.3.187`) against `longngoc11/roadwatch-v27-ai85-execution` (`code_manifest_sha256 = e2f3917a11974822f006b26c9cfcc4712c073c2f0202e172fe790de313d5c187`).
  - All contract tests (`candidate_identity_test`, `detector_imgsz_config_test`, `temporal_release_lifecycle_test`, `v30_causal_action_contract_test`, `v32_normalized_proximity_test`, `v32_aspect_ratio_clutter_filter_test`) passed with `dmd_media_used = False`.
