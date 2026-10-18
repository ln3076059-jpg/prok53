# 05. Recovery of Historical Training Metrics
From `models/recovery_extracted/MULTIMODEL_V2_RTX5060TI_16GB_ALL_PRETRAIN_PENDING_TRAIN_MANIFEST.json`:
- **Phone Detector:**
  - mAP50(B): **94.48%**
  - mAP50-95(B): **70.03%**
  - Precision: **97.13%**
  - Recall: **82.73%**
- **Seatbelt Detector:**
  - mAP50(B): **94.73%**
  - mAP50-95(B): **53.70%**
  - Precision: **92.57%**
  - Recall: **87.85%**
- **Seatbelt Classifier:**
  - Top-1 Accuracy: **80.97%**
  - Top-5 Accuracy: **100.00%**
These high validation metrics confirm the object detectors are well-trained. Low system Event Recall is caused by pipeline downscaling and event gating bottlenecks, NOT underfitted detectors.
