# ROADWATCH V19 — Checkpoint Inventory & Provenance Verification
## 1. Verified Model Files
Audit of `models/` verified **59 checkpoint and configuration files**:
- `models/active/v2/phone_detector.pt` (19.2 MB, SHA256: `840a29cb...`, YOLO11s)
  - Historical mAP50: **94.48%**, mAP50-95: **70.03%**, Precision: **97.13%**, Recall: **82.73%**.
- `models/active/v2/seatbelt_detector.pt` (19.3 MB, SHA256: `361436ab...`, YOLO11s)
  - Historical mAP50: **94.73%**, mAP50-95: **53.70%**, Precision: **92.57%**, Recall: **87.85%**.
- `models/active/v2/seatbelt_classifier.pt` (11.0 MB, SHA256: `e55158e3...`, YOLO11s-cls)
  - Historical Top-1 Accuracy: **80.97%**, Top-5: **100.00%**.
- `models/auxiliary/yolo11n.pt` (5.6 MB, SHA256: `0ebbc80d...`, YOLO11n COCO)
- `models/auxiliary/yolo11n-pose.pt` (6.2 MB, SHA256: `869e83fc...`, YOLO11n-Pose)

## 2. Checkpoint Contract Status
| Model Role | Architecture | Verified SHA-256 | Runnable | Provenance Status |
| :--- | :--- | :--- | :--- | :--- |
| Phone Specialist | YOLO11s | 840a29cb21... | Yes | RECOVERED_TRAINING_HISTORY |
| Seatbelt Detector | YOLO11s | 361436ab07... | Yes | RECOVERED_TRAINING_HISTORY |
| Seatbelt Classifier | YOLO11s-cls | e55158e3f1... | Yes | RECOVERED_TRAINING_HISTORY |
| Auxiliary Object | YOLO11n | 0ebbc80d4a... | Yes | AUXILIARY_PRETRAINED |
| Auxiliary Pose | YOLO11n-Pose | 869e83fcdff...| Yes | AUXILIARY_PRETRAINED |
| Action LOSO V5/V6 | ResNet18-GRU | (recovery dir) | Offline | HISTORICAL_EXTRACTED |
| OnlineE3 Head | NumericHead | N/A | Missing | NOT_FOUND_LOCALLY |
