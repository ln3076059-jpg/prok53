# 06. Behavior ROI Cropping & Hand/Face Localization

## 1. Spatial Formulation
Rather than processing whole cabin images, localized behavior extraction focuses computational capacity on relevant anatomical interaction zones:
* **Driver Upper Body ROI:** `[x1, y1, x2, y2]` dynamic torso bounding box.
* **Hand Interaction Zones:** 1.5x expansion around left and right wrists predicted by YOLO11n-pose.
* **Head / Ear Proximity Zone:** 1.3x expansion around ear/face keypoints to isolate calling actions.

## 2. Coordinate Transformation Contract
All sub-crops must satisfy the invertibility guarantee:
$$\mathbf{b}_{\text{original}} = \mathbf{b}_{\text{local}} + \mathbf{o}_{\text{crop}}$$
Implemented in `training/dmd/autonomous_phone.py` via `local_to_original()` and validated in unit tests to guarantee zero coordinate drift during NMS fusion.
