# 09. Detector & SAHI Ablation Experiments
On gA-1 s2 frame at 60s (ground truth `phonecall_right`):
- **A0 (Specialist, Full Frame 640x640):** 0 detections.
- **A1 (Auxiliary COCO, Full Frame 640x640):** 0 detections.
- **A2 (Specialist, Driver ROI Crop):** 0 detections.
- **A3 (Auxiliary COCO, Driver ROI Crop):** **1 phone detected!**
  - Coordinates: `[406.5, 91.9, 446.8, 146.3]`, Conf: 0.12.
  - Exactly matches driver ear position!
Conclusion: Sourced-resolution cropping/tiled inference directly recovers missed phone proposals without retraining.
