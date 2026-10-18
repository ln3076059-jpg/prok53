# ROADWATCH V20 — Scientific Integrity Audit of V19 Replays
## 1. Executive Summary & Audit Findings
A thorough line-by-line audit of `training/dmd/v19_bug_fix_replay.py` and V19 deliverables was conducted to verify scientific validity. The audit confirmed several critical methodological concerns:

### P0.1: Ground-Truth Selected Proposal Leakage in R4/R5
- **Finding:** In `v19_bug_fix_replay.py`, `proposal_injections` was loaded from `reports/v9_research/phone_proposal_mining.json`.
- **Evidence:** That mining file was generated exclusively on the 13 historical false negative cases (`gB-9-phone-01..09`, `gZ-36-phone-01..05`, `gC-14-phone-02..06`).
- **Code Line:** Line 105 in `v19_bug_fix_replay.py`:
  `if abs(obs.timestamp - p_time) <= 0.5: if p_conf > conf: conf = p_conf; ctx = "HANDHELD_USE"`
- **Severity: CRITICAL.** Proposals were injected *only* at timestamps where known ground-truth phone interactions occurred. Negative intervals across the video were never evaluated for auxiliary proposals. This artificially boosted TP without testing false alarm rates on negative intervals.
- **RULING:** **R4 and R5 are RECLASSIFIED as `GT-INFORMED_DEVELOPMENT_DIAGNOSTIC`**. They cannot be claimed as autonomous model accuracy.

### P0.2: Injected Handheld Context Override
- **Finding:** In V19 replay, whenever an injected proposal had confidence above the threshold, the code forcibly set `ctx = "HANDHELD_USE"`.
- **Evidence:** A phone bounding box alone does not prove the phone is in active handheld use. Real evidence must come from verified spatial proximity to hands and face (`classify_phone_context`). Setting `ctx = "HANDHELD_USE"` bypassed context gating.
- **RULING:** In V20 autonomous inference, phone context is strictly evaluated from spatial keypoint proximity or remains `UNKNOWN`.

### P0.3: Fabricated Driver Role Evidence
- **Finding:** In V19 replay, observations were instantiated with `occupant_role = "driver"` and `occupant_role_confidence = 1.0` hardcoded.
- **Evidence:** Real autonomous inference must verify occupant track identity and seat location. Unverified tracks must remain `UNKNOWN` or `NEEDS_REVIEW`.

### P0.4: Replay Scope Reclassification
The Roadwatch evaluation ladder is now strictly classified into four distinct levels:
1. `LEVEL 1 — SYNTHETIC_UNIT_TEST`: Tests on synthetic fixtures.
2. `LEVEL 2 — GT_INFORMED_DIAGNOSTIC`: Replays using oracle or FN-selected timestamps (e.g. V19 R4/R5).
3. `LEVEL 3 — AUTONOMOUS_DEVELOPMENT_REPLAY`: End-to-end inference from video pixels across continuous positive and negative intervals without GT.
4. `LEVEL 4 — LOCKED_INDEPENDENT_HOLDOUT`: Single-shot evaluation on untouched multi-subject panels (>=5 subjects).

### P0.6: R4 vs R5 Accuracy Claim
- **Finding:** In V19, R4 and R5 produced identical numbers: TP=7, FP=4, FN=13, F1=45.16%.
- **RULING:** R5 did not improve accuracy over R4. Claiming improvement between R4 and R5 was unsupported.
