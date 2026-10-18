# ROADWATCH V20 — Final Senior AI Technical Committee Decision
## 1. Official Verdict
- **Production Activation:** `production_approved = false` (MAINTAINED).
- **AI85 Certification:** `INDEPENDENT_AI85_NOT_ESTABLISHED`.

## 2. Justification
1. **Audit Finding:** V19 R4/R5 suffered evaluation leakage by injecting proposals derived exclusively from known FN intervals. When reclassified as a diagnostic, autonomous performance on development cohort remains below the 85% target.
2. **Missing Action Weights:** Compatible fitted `OnlineE3` numeric head is absent locally.
3. **Single-Subject Holdout Limitation:** DMD `gA-1` has frame-level diagnostic exposure and cannot certify multi-subject generalizability.
4. **Seatbelt Label Dependency:** 500 candidate labels require completed human validation before retraining.

## 3. High-Value Next Engineering Steps
1. Deploy Native Driver Crop detection into the production pipeline.
2. Complete human review of the 500 seatbelt candidates and train a torso-attention classifier.
3. Lock a clean, multi-subject independent test panel (>=5 unseen drivers) for single-shot AI85+ certification.
