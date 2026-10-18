# ROADWATCH V19 — Phone Detector & SAHI / Native Crop Ablation
## 1. Experimental Evidence
- **Full-Frame Baseline (1280x720 -> 640x640):** Yielded 0 detections on `gB-9` and 0 detections on `gA-1` frame 60.
- **Native Resolution Driver Crop (YOLO11n Auxiliary):** Successfully detected the phone at ear position `[406.5, 91.9, 446.8, 146.3]` on `gA-1` frame 60 (confidence 0.12).
- **Conclusion:** Cropping the driver region prior to inference solves the small-phone resolution bottleneck.
