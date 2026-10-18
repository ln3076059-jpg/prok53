# ROADWATCH V20 — Existing Checkpoint Recovery & Provenance
## 1. Verified Active Models
- `phone_detector.pt` (YOLO11s, SHA256: `840a29cb...`): Historical mAP50 = 94.48%.
- `seatbelt_detector.pt` (YOLO11s, SHA256: `361436ab...`): Historical mAP50 = 94.73%.
- `seatbelt_classifier.pt` (YOLO11s-cls, SHA256: `e55158e3...`): Historical Top-1 = 80.97%.
- `yolo11n.pt` (COCO, SHA256: `0ebbc80d...`): Auxiliary object detector.
- `yolo11n-pose.pt` (Pose, SHA256: `869e83fc...`): Auxiliary pose estimator.

## 2. Missing Artifacts
- `OnlineE3` fitted JSON head (`NumericHead`): Does not exist on filesystem. Synthetic dummy weights are strictly restricted to unit tests and never reported as real accuracy.
