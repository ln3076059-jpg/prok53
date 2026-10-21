# ROADWATCH V24 — Comprehensive Source & Evidence Audit

**Document ID:** `RW-V24-AUDIT-20261020`  
**Date:** 2026-10-20  
**Evaluator Council:** Principal AI Scientist, Senior CV & Action Recognition Engineers, MLOps Lead  
**Target Repository:** `D:\.idea\giangdoantotnghiep`  
**Git HEAD:** `98502a4ab740d3ede3df18a6e217fb1861c009e9` (Branch: `roadwatch-v4-candidate`)  

---

## 1. Executive Summary of Audit Findings

| Issue ID | Severity | Classification | Component | Primary File & Line | Summary |
|---|---|---|---|---|---|
| **P0-A** | CRITICAL | `CONFIRMED_CODE_DEFECT` | Driver Association | `training/dmd/run_verified_v23.py:225-242, 402` | Role inferred from `poses[0]` without seat geometry test; fallback box `Box(0,0,832,612)`; `"VERIFIED" in status` substring bug matches `"UNVERIFIED"`. |
| **P0-B** | CRITICAL | `CONFIRMED_CODE_DEFECT` | Temporal Latency & Alignment | `training/dmd/run_verified_v23.py:194, 375` | Feature lookup in `TimestampEmbeddingIndex` allows future feature leakage ($\pm 0.25\text{s}$); centered action score is fused into current-time observation rather than window center observation. |
| **P0-C** | CRITICAL | `CONFIRMED_CODE_DEFECT` | Temporal Action History | `training/dmd/run_verified_v23.py:296` | `observe_embedding()` only called when `top_candidate is not None`. Gaps between candidate detections (> 0.75s) trigger `self.reset()`, purging 9-frame buffer and preventing deep action execution across 89% of frames. |
| **P0-D** | HIGH | `CONFIRMED_CODE_DEFECT` | Action Adapter API & Fusion | `training/dmd/run_verified_v23.py:308-340, 370` | Direct private model invocation (`adapter._deep_model`) bypassing `predict()`; action evaluated on `top_candidate` but observation synthesized from `best_associated` (occupant mismatch). |
| **P0-E** | HIGH | `MISSING_EVIDENCE` | Provenance & Scope | `training/dmd/run_verified_v23.py:136-157` | Hardcoded default cohort metrics; cached trace replay wall times (~3s) reported without separating `CACHED_REPLAY_DIAGNOSTIC` from `FULL_VIDEO_AUTONOMOUS`. |
| **P1-A** | MEDIUM | `CONFIRMED_CODE_DEFECT` | Event Boundary Tracking | `training/dmd/v22_boundary_diagnostic.py:68` | `ActiveBoundaryReporter` extends active intervals using `observation.confidence` instead of `observation.fusion_score`, decoupling fused action decisions from temporal output. |
| **P1-B** | MEDIUM | `MODEL_LIMITATION` | Seatbelt Classifier 3-Class | `backend/ai/auxiliary.py:SeatbeltClassifier` | `uncertain_or_occluded` recall is 4.0% (1 of 25) with 10.29% false unfastened rate. Architecture lacks torso crop and class-balanced focal weights. |

---

## 2. In-Depth Defect Breakdown

### P0-A. Association V23 Infers Role from Pose without Geometry Verification

- **File / Function / Line:** `training/dmd/run_verified_v23.py`, `replay_v23_record()`, lines 225–243, line 273, line 402.
- **Concrete Code Evidence:**
  ```python
  # Line 225-227
  driver_seat = calibration["seats"][0]
  poses = record.get("poses", [])
  pose = PoseEvidence(**poses[0]) if poses else None

  # Line 232-241
  if pose is not None and pose.confidence >= 0.25:
      p_box = Box(*pose.xyxy) if pose.xyxy else Box(0, 0, 832, 612)
      driver_track = OccupantTrack(
          track_id="driver_0",
          role="driver",
          box=p_box,
          verified=True,
          confidence=float(pose.confidence),
      )
  # Line 273
  context_verified=bool(occupants)

  # Line 402
  is_verified_seat = "VERIFIED" in calibration.get("status", "")
  ```
- **Technical Impact:**
  1. `poses[0]` is blindly chosen. If a passenger pose is listed first, the passenger is assigned `track_id="driver_0"` and `role="driver"`.
  2. If `pose.xyxy` is None/empty, a synthetic bounding box `Box(0, 0, 832, 612)` is fabricated rather than failing closed.
  3. Pose bounding box in driver crop coordinates is never transformed to full-frame pixel coordinates `(1280x720)` before checking intersection with calibrated seat geometries.
  4. `context_verified=bool(occupants)` equates the mere existence of an occupant track with physical verification of phone context (handheld/mounted).
  5. `"VERIFIED" in calibration.get("status", "")` is a critical Python substring flaw: if status is `"UNVERIFIED_CALIBRATION"`, the expression evaluates to `True`!
- **Minimal Provable Fix:**
  1. Validate camera calibration strictly: check camera ID (`BODY`), resolution (`1280x720`), vehicle configuration, and exact status `status in {"VERIFIED_STATIC_CALIBRATION", "VERIFIED_PHYSICAL_CALIBRATION"} and not status.startswith("UNVERIFIED")`.
  2. Transform pose coordinates from ROI local `(x, y)` to full frame `(x + roi_x1, y + roi_y1)`.
  3. Iterate over ALL detected poses in `record["poses"]`. Compute geometric intersection over bounding box with all calibrated seats in `calibration["seats"]`. Assign role (`driver`, `front_passenger`, `unknown`) based on maximum geometric overlap $\ge \text{min\_overlap} (0.35)$.
  4. Fail closed when pose or bbox is missing: do not fabricate `Box(0, 0, 832, 612)`.
  5. Decouple `context_verified` from `bool(occupants)`. Phone context verification requires proximity/mount evidence, not occupant presence.
- **Verification Test:** Unit test in `tests/test_v24_recovery.py`: passenger-first pose order, missing bbox, swapped pose order, `"UNVERIFIED"` calibration status string.

---

### P0-B. Temporal Latency Contract & Future Feature Leakage

- **File / Function / Line:** `training/dmd/run_verified_v23.py`, `TimestampEmbeddingIndex.get_embedding()`, lines 178–198; `replay_v23_record()`, lines 371–390.
- **Concrete Code Evidence:**
  ```python
  # Line 194-197
  idx = int(np.argmin(np.abs(self.timestamps - timestamp)))
  if abs(self.timestamps[idx] - timestamp) <= max_gap:
      return self.embeddings[idx]
  ```
- **Technical Impact:**
  1. `np.argmin(np.abs(self.timestamps - timestamp))` can select an embedding where `self.timestamps[idx] > timestamp` (up to $+0.25\text{s}$ into the future). In an online causal pipeline, future frames are physically impossible to observe.
  2. The V6 action model was trained with `label_position = "CENTER"` over a 9-frame window with $0.504\text{s}$ sample period (declared latency $= 4 \times 0.504\text{s} \approx 2.016\text{s}$). When the 9th frame arrives at time $t_8$, the model predicts the state at window center $t_4 = t_8 - 2.016\text{s}$.
  3. V23 fused this score directly into observation at $t_8$! This backdated future actions or applied past actions to present detector observations.
- **Minimal Provable Fix:**
  1. Define explicit temporal contract: `source_frame_timestamp`, `feature_timestamp`, `window_timestamps`, `score_timestamp`, `available_at_timestamp`.
  2. In strict causal online mode: enforce `self.timestamps[idx] <= timestamp` and `timestamp - self.timestamps[idx] <= max_gap`.
  3. In `DECLARED_LATENCY_CENTERED` mode: hold an observation queue. When the 9-frame window completes at $t_8$, compute the centered action score for target time $t_4 = t_8 - 2.016\text{s}$. Fuse the action score with the queue observation at $t_4$, emitting the decision with availability timestamp $t_8$.
- **Verification Test:** Unit test with non-monotonic timestamps, time gaps, and verification that future signals do not alter past decisions before availability.

---

### P0-C. Action History Dependently Gated on Phone Candidates

- **File / Function / Line:** `training/dmd/run_verified_v23.py`, lines 296–308; `training/dmd/action_adapter.py`, lines 187–203.
- **Concrete Code Evidence:**
  ```python
  # run_verified_v23.py line 296
  if adapter is not None and top_candidate is not None:
      t_sec = float(record["timestamp"])
      emb = embedding_index.get_embedding(t_sec) if embedding_index else None
      if emb is not None and adapter._deep_model is not None:
          seq = adapter.observe_embedding(emb, (subject_id, session_id, camera_id, "driver_0"), t_sec)

  # action_adapter.py lines 194-199
  if stream_key != self._stream_key or (
      self._last_timestamp is not None
      and (timestamp <= self._last_timestamp or timestamp - self._last_timestamp > 1.5 * period)
  ):
      self.reset()
  ```
- **Technical Impact:**
  1. On safe driving frames or intervals without phone candidate boxes, `observe_embedding` was completely bypassed.
  2. Because candidate detections were sporadic, consecutive gaps $> 0.75\text{s}$ triggered `self.reset()`, wiping the 9-frame buffer.
  3. As proven by raw prediction logs, the deep model executed only 176 forwards on `gC-14` and 25 forwards on `gZ-36` (total 201). Over 89% of valid cabin frames had zero deep action coverage!
- **Minimal Provable Fix:**
  1. Continuously feed embeddings on every valid occupant frame at the sampling cadence ($0.504\text{s}$), independent of phone detection candidate presence.
  2. Maintain a sliding 9-frame embedding history for each verified occupant track.
  3. When candidate evaluations occur, query the continuously maintained temporal action score corresponding to that occupant and timestamp.
- **Verification Test:** Test confirming continuous buffer accumulation and forward execution across continuous safe driving video frames.

---

### P0-D. Runner Bypasses Adapter API and Fuses Inconsistent Candidates

- **File / Function / Line:** `training/dmd/run_verified_v23.py`, lines 308–341, 370.
- **Concrete Code Evidence:**
  ```python
  # Lines 308-320: Private model bypass
  with torch.inference_mode():
      coarse_logits, phone_logits = adapter._deep_model(seq_tensor)

  # Line 324: Action evaluated on top_candidate
  pose_feat = adapter._extract_pose_feature_vector(pose, top_candidate["local_box"])

  # Line 370: Observation synthesized from best_associated
  if best_associated is not None and not action_suppressed:
      det_conf = float(best_associated["confidence"])
  ```
- **Technical Impact:**
  1. Private adapter bypass violates encapsulation and causes divergent preprocessing logic between research runners and backend services.
  2. `top_candidate` is simply the detection with the highest confidence across all proposals (potentially belonging to a front passenger). `best_associated` is the highest-confidence driver detection. When both exist, action features were extracted from the passenger phone box, but fused into the driver detection!
- **Minimal Provable Fix:**
  1. Route all action inference through unified public adapter method `ActionInferenceAdapter.predict_from_embedding()` or unified `predict()`.
  2. Ensure that action features, pose keypoints, and detection candidates belong to the EXACT same occupant track and timestamp.
- **Verification Test:** Unit test with two simultaneous detections (passenger high-conf phone, driver moderate-conf phone) confirming that driver action reasoning receives driver phone coordinates.

---

### P0-E. Replay, Cache, and Provenance Scope Separation

- **File / Function / Line:** `training/dmd/run_verified_v23.py`, lines 136–157, 452–486.
- **Technical Impact:**
  1. `load_cohort()` hardcoded 20 GT and 15.27 minutes without cryptographically validating the underlying OpenLABEL annotations and video files.
  2. The runner read precomputed trace JSONL files and NPZ embeddings, completing in ~3 seconds. Reporting this without explicit scope labeling (`CACHED_REPLAY_DIAGNOSTIC` vs `FULL_VIDEO_AUTONOMOUS`) obscured real-world inference throughput.
- **Minimal Provable Fix:**
  1. Compute SHA256 hashes of all input files (traces, NPZ caches, OpenLABEL annotations).
  2. Explicitly label each experiment run with its execution scope: `CACHED_REPLAY_DIAGNOSTIC` or `FULL_VIDEO_AUTONOMOUS`.
  3. Include a full end-to-end video pipeline candidate (decoding + detection + pose + action) to measure actual FPS, latency, and memory footprint.
- **Verification Test:** Automated manifest validation checking SHA256 of all input assets.
