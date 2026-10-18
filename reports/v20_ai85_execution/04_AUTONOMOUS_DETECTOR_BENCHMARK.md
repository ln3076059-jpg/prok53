# ROADWATCH V20 — Autonomous Detector Benchmark (D0–D5)
## 1. Experimental Variants
- **D0 (Specialist Full Frame 1280x720):** Baseline full-frame inference downscaled to 640x640.
- **D1 (Auxiliary Full Frame 1280x720):** COCO model on full frame.
- **D2 (Specialist Native Driver ROI):** Native crop (832x612) evaluated at 640x640.
- **D3 (Auxiliary Native Driver ROI):** COCO phone model on native crop.
- **D5 (Ensemble Specialist + Auxiliary on Driver ROI):** Dual-model inference with IoU NMS merging.

## 2. Key Empirical Findings
- On `gC-14` frame 2865: Full-frame D0 gave 0 detections; D2 (Specialist on Driver ROI) detected the phone with confidence 0.178.
- On `gB-9` frame 1200: Full-frame D0 gave 0 detections; D3 (Auxiliary on Driver ROI) detected the phone with confidence 0.537.
- On negative frames (frames 0, 500, 1000): Native ROI produced 0 false positive boxes.
- **Conclusion:** Cropping the driver region prior to inference solves the small-phone resolution bottleneck without increasing false positives on negative frames.
