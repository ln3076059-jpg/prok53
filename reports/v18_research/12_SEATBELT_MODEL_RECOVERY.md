# 12. Seatbelt Model Recovery & Status
- **Seatbelt Detector:** YOLO11s (`mAP50 = 94.73%`, Precision `92.57%`, Recall `87.85%`).
- **Seatbelt Classifier:** YOLO11s-cls (`Accuracy Top-1 = 80.97%`).
- **Governance:** 500 samples in review worklist remain pending human verification.
- **Policy:** UNCERTAIN_OR_OCCLUDED is preserved as a separate class; never forced to UNFASTENED.
