# 13. Three-State Seatbelt Classification & Human Review

## 1. Three-State Taxonomy
To eliminate false unfastened alerts:
1. `FASTENED`: Diagonal belt clearly visible across torso.
2. `UNFASTENED`: Torso fully visible, absence of belt confirmed.
3. `UNCERTAIN_OR_OCCLUDED`: Dark clothing, heavy jacket, bag strap, or poor lighting obscures torso.

## 2. The 500 Pending Candidates Worklist
500 images reside in `dataset/review_worklists/v2_pending_approval/05_seatbelt_uncertain_classifier_candidates`.
* **Policy:** Zero automatic approval by AI. Every sample requires explicit human confirmation.
* **Progression S0-S5:** S0 (baseline) -> S1 (expanded torso crop) -> S2 (belt presence head) -> S3 (triplet attention) -> S4 (hard negatives) -> S5 (temperature scaling calibration).
