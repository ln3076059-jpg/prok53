# V2 Phone Detector Calibration

## 1. Overview
- **Component:** Phone Detector (YOLO11s)
- **Model SHA:** `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`
- **Calibration Split:** `datasets/derived/v2_pretrain_pending_approval/phone_detector/images/val` (302 images)
- **Methodology:** Validation-only confidence sweep.

## 2. Threshold Selection
A confidence sweep was performed on the canonical validation set. The F1 curve indicates that an optimal balance between precision and recall is achieved at **confidence = 0.5996**.
- At Conf = 0.5996, the model yields **Precision 96.67%**, **Recall 84.17%**, and **Max F1 89.99%**.
- NMS IOU is kept at the default 0.7.

## 3. Justification
Phone detection feeds a downstream policy. False positive bounding boxes (e.g., detecting hands as phones) can lead to false violations if not smoothed properly. A confidence threshold of 0.60 maintains a high precision (96.67%) while retaining a very strong recall (84.17%). This heavily suppresses background noise and ensures that only highly confident phone detections enter the temporal smoothing phase.
