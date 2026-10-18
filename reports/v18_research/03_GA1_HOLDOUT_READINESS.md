# 03. gA-1 Holdout Protection & Readiness
- **Subject Identity:** Subject ID 1, Group `gA`.
- **Contamination Check:**
  - `gA-1` does NOT appear in any training splits (V2, V5, V6, V7).
  - `gA-1` does NOT appear in historical validation sets (`gB-9`, `gC-14`, `gZ-36`).
  - `gA-1` was NEVER used for hyperparameter tuning or threshold selection.
- **Lock State:** `LOCKED_INDEPENDENT_HOLDOUT`. Added to `PROTECTED_SUBJECTS` in `training/data_governance.py`.
- **Integrity Rule:** `gA-1` is strictly reserved for one-time independent generalization testing; training or fitting on `gA-1` is strictly blocked.
