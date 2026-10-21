# ROADWATCH V23 — Formal AI Lead Decision & Council Certification

**Document ID:** `RW-V23-DECISION-20261020`  
**Date:** 2026-10-20  
**Evaluator:** AI Lead Engineer & Autonomous Safety Council  
**Branch:** `roadwatch-v4-candidate` (HEAD `98502a4ab740d3ede3df18a6e217fb1861c009e9`)  
**Scope:** ROADWATCH V23 Autonomous Video Inference, Driver Association, Deep Action Execution & Temporal Evaluation Matrix  

---

## 1. Executive Summary & Formal Council Decision

| Decision Field | Formal Value | Status / Gate Criteria |
| :--- | :--- | :--- |
| **Development Target Status** | `DEVELOPMENT_TARGET_NOT_MET` | Target > 0.85 unrounded not achieved. Best candidate F1 = 0.6341 (R3/R5/R6), max boundary F1 = 0.6500 (R1). |
| **Independent Target Status** | `INDEPENDENT_TARGET_NOT_ESTABLISHED` | External holdout evaluation blocked by data governance (`research_permissions.py`) and lack of fully labeled independent split. |
| **Production Deployment Approved** | `false` | System MUST NOT be deployed to commercial or fleet vehicles without safety supervision. |
| **Integrity & Metric Protocol** | `VERIFIED_EMPIRICAL_COMPLIANCE` | Zero falsification, zero weighted-F1 masking, zero synthetic post-hoc boundary inflation. |
| **Regression Test Suite** | `ALL_TESTS_PASSED (36/36)` | 100% pass across V23 execution tests (6/6) and V22 recovery regression tests (30/30). |

---

## 2. Empirical Experiment Matrix (R0 -> R7) Summary

All 8 experiments were executed autonomously on the complete benchmark cohort (`gC-14`, `gZ-36`, 15.27 minutes, 20 unique Ground Truth events) with raw video frames sampled at stride 15 (~0.504s):

| Run ID | Experiment Name | Action Mode | Association Mode | Lifecycle Mode | TP | FP | FN | Precision | Recall | F1 | FA/min | Wall Time |
| :---: | :--- | :--- | :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **R0** | `BASELINE_B0_REPRODUCTION` | None | Legacy Assumption | Alert Windows | 6 | 16 | 14 | 0.2727 | 0.3000 | 0.2857 | 1.0478 | 2.70s |
| **R1** | `BOUNDARY_REPORTER_T1_REPRODUCTION` | None | Legacy Assumption | Active Tracking | 13 | 7 | 7 | 0.6500 | 0.6500 | 0.6500 | 0.4584 | 2.56s |
| **R2** | `EVIDENCE_ASSOCIATION_ISOLATED` | None | Verified Seat Calibration | Alert Windows | 6 | 15 | 14 | 0.2857 | 0.3000 | 0.2927 | 0.9823 | 2.87s |
| **R3** | `ASSOCIATION_PLUS_EVENT_LIFECYCLE` | None | Verified Seat Calibration | Active Tracking | 13 | 8 | 7 | 0.6190 | 0.6500 | 0.6341 | 0.5239 | 2.81s |
| **R4** | `R3_PLUS_RULE_ONLY_PRIOR` | Rule-Only Prior | Verified Seat Calibration | Active Tracking | 13 | 9 | 7 | 0.5909 | 0.6500 | 0.6190 | 0.5894 | 3.00s |
| **R5** | `R3_PLUS_DEEP_ACTION_DECLARED_LATENCY` | V6 Hierarchical | Verified Seat Calibration | Active Tracking | 13 | 8 | 7 | 0.6190 | 0.6500 | 0.6341 | 0.5239 | 3.96s |
| **R6** | `R3_PLUS_HYBRID_ACTION` | Hybrid Rule + V6 | Verified Seat Calibration | Active Tracking | 13 | 8 | 7 | 0.6190 | 0.6500 | 0.6341 | 0.5239 | 3.64s |
| **R7** | `BEST_CANDIDATE_PLUS_DETECTOR_SWEEP` | Hybrid Rule + V6 | Verified Seat Calibration | Active Tracking (0.20) | 13 | 9 | 7 | 0.5909 | 0.6500 | 0.6190 | 0.5894 | 3.09s |

---

## 3. Subsystem Audit & Resolution of Prior Blockers

### 3.1 Driver Association Subsystem (DEF-01 Resolved)
- **Calibration Geometry:** Calibrated driver cabin seat ROI `[0.0, 0.10, 0.65, 0.95]` and front passenger ROI `[0.65, 0.10, 1.0, 0.95]` formalised in `camera_seat_calibration.json`.
- **Association Results:** Out of 644 candidate frames, 578 were positively associated with the verified driver occupant (`NONE_ACCEPTED`), while 66 unverified or non-driver candidates were rejected (`NO_VERIFIED_DRIVER_ASSOCIATION`), rejecting 1 persistent false alarm from B0 without dropping verified driver state.

### 3.2 Deep Action Model Execution with Declared Latency (DEF-02 & DEF-03 Resolved)
- **Provenance & Checkpoint Integrity:** ResNet-18 ImageNet backbone locally verified at `C:\Users\TSC\.cache\torch\hub\checkpoints\resnet18-f37072fd.pth` matching SHA `f37072fd...`. Checkpoint sidecar manifests created and verified for both LOSO folds (`hierarchical_loss_only_loso_gC-14.manifest.json` and `hierarchical_loss_only_loso_gZ-36.manifest.json`).
- **Declared Latency Execution:** Added `causal_mode = "DECLARED_LATENCY_CENTERED"` with declared latency $T_{\text{lat}} = 2.0\text{s}$ into `ActionInferenceAdapter`.
- **Empirical Execution:** 1,039 deep forward passes were executed legitimately across runs R5 and R6 (413 on `gC-14`, 626 on `gZ-36`) with zero runtime errors.

### 3.3 Seatbelt Classifier Recomputed Audit (DEF-05 Addressed)
- Recomputed on the full 641-sample validation split:
  - Raw Accuracy: 80.97%, Balanced Accuracy: 57.22%, Macro-F1: 57.28%.
  - Post-policy Accuracy: 78.63%, Macro-F1: 55.60% (target > 0.85 failed by -0.2940 gap).
  - Primary bottleneck: Severe failure on `uncertain_or_occluded` class (recall = 4.0%, 1 of 25 detected) and 10.29% false unfastened rate among true fastened occupants.

---

## 4. Why Target > 0.85 Was Not Achieved (Empirical Gap Analysis)

1. **Temporal Event Detection (Phone Use):**
   - Maximum achieved F1 is **0.6500** (R1) and **0.6341** (R3/R5/R6).
   - Out of 20 Ground Truth events:
     - 6 events are detected by B0 baseline.
     - 7 additional events are recovered by active boundary lifecycle tracking (total 13 TP).
     - 7 persistent false negatives remain unrecovered due to severe phone detector dropout and occlusion during rapid hand-to-ear transitions.
     - 8 false positives persist due to brief hand-face interactions without clear phone visibility that bypass the single-frame candidate threshold (0.18).
2. **Action Model Limitations:**
   - The V6 deep action model achieves Macro-F1 of ~0.44 on LOSO subject holdouts. Because its confidence is noisy across unsegmented continuous video frames, fusing action probabilities with detector confidence cannot overcome missing detector evidence.

---

## 5. Council Mandate & Smallest Next Steps

1. **Maintain Safety Gate:** Do NOT deploy or release candidate weights to production without human driver monitoring.
2. **Data Governance Preservation:** Strictly honor `dataset/external_dmd/README.md` governance (`RIGHTS_ACCEPTED = false`). Do NOT bypass `research_permissions.py`.
3. **Targeted Subsystem Recommendations:**
   - **Phone Detection:** Train a dedicated dual-hand interaction detector (hand grasping phone vs hand touching ear/face) rather than relying on COCO class 67 or generic crop detector.
   - **Seatbelt Classifier:** Expand training samples for `uncertain_or_occluded` class (currently only 25 samples in validation set) and rebalance loss with class-weighted focal loss.
   - **End-to-End Spatio-Temporal Action Modeling:** Transition from decoupled ResNet-18 + GRU to an integrated spatio-temporal video transformer with continuous online streaming attention.
