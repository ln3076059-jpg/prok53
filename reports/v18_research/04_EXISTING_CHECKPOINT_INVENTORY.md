# 04. Existing Checkpoint Inventory
The project contains 59 model files in `models/`:
1. `models/active/v2/phone_detector.pt` (18.32 MB, SHA-256 `840a29cb...`): YOLO11s trained on cabin phone dataset.
2. `models/active/v2/seatbelt_detector.pt` (18.45 MB, SHA-256 `361436ab...`): YOLO11s trained on seatbelt bounding boxes.
3. `models/active/v2/seatbelt_classifier.pt` (10.52 MB, SHA-256 `e55158e3...`): YOLO11s-cls 3-class classifier.
4. `models/auxiliary/yolo11n.pt` (5.35 MB): COCO general object detector (cell phone class 67).
5. `models/auxiliary/yolo11n-pose.pt` (5.97 MB): Keypoint pose estimator for driver hands and wrists.
6. Research series: V5 ResNet18-GRU (3 LOSO checkpoints), V5 Tiny Temporal CNN, V6 Action Ablation (12 LOSO checkpoints), V7 Optimization (12 LOSO checkpoints), AI85 upgrade (9 LOSO checkpoints).
