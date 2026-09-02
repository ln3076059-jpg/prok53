# V2 Kaggle Handoff
1. Create PRIVATE Kaggle Dataset.
2. Upload current V2 bundle (`MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL_PORTABLE.zip`).
3. Upload notebook (`kaggle_train_v2_pretrain_pending_approval.ipynb`) or push kernel.
4. Attach dataset.
5. Enable GPU.
6. Initially leave `PREP_ONLY=True`, `START_TRAINING=False`.
7. Run All.
8. Confirm KAGGLE_PREFLIGHT=PASS.
9. Set `RUN_SMOKE=True`.
10. Run one-epoch smoke.
11. Only after smoke PASS: `START_TRAINING=True`.
12. Run full training later.
13. Save/download recovery artifacts (`V2_PRETRAIN_RECOVERY.zip`).
