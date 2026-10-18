# ROADWATCH V19 — Final Technical Committee Decision
## 1. Official Decision
- **Production Approval:** `production_approved = false` (MAINTAINED).
- **Scientific Certification:** `INDEPENDENT_AI85_NOT_ESTABLISHED`.

## 2. Rationale
1. **Event Recall Gap:** Replay R5 achieves Event Recall 35.00% on the 2-subject development cohort (target >85.00%).
2. **Missing Action Artifact:** Compatible fitted `OnlineE3` numeric head is absent locally.
3. **Holdout Limitation:** DMD `gA-1` has frame-level diagnostic exposure and cannot certify multi-subject generalizability.
4. **Seatbelt Label Dependency:** 500 candidate labels require completed human validation.

## 3. High-Value Next Engineering Steps
1. Deploy Driver-Crop SAHI / high-resolution proposals into the active pipeline to permanently resolve the downscaling bottleneck.
2. Complete human review of the 500 seatbelt candidates and train a torso-attention classifier.
3. Acquire an untouched multi-subject test panel (>=5 drivers) with locked annotations for definitive independent AI85+ validation.
