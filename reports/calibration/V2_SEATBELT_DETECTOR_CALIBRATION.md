# V2 Seatbelt Detector Calibration

## 1. Overview
- **Component:** Seatbelt Detector (YOLO11s)
- **Model SHA:** `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500`
- **Calibration Split:** `datasets/derived/v2_pretrain_pending_approval/seatbelt_detector/images/val` (636 images)
- **Methodology:** Validation-only confidence sweep.

## 2. Threshold Selection
A confidence sweep was performed on the canonical validation set. The F1 curve indicates that an optimal balance between precision and recall is achieved at **confidence = 0.4074**.
- At Conf = 0.4074, the model yields **Precision 93.29%**, **Recall 87.12%**, and **Max F1 90.10%**.

## 3. Justification
The seatbelt detector is strictly an `occupant_upper_body` localizer. Missing an occupant's upper body (False Negative) means the downstream classifier won't run, which might miss a seatbelt violation. Conversely, false positive crops might pass non-occupant areas to the classifier, but the classifier is trained to reject them or map them to uncertain. Therefore, maximizing recall is slightly more critical here, making Conf=0.40 a safe and effective threshold. A missing belt detection DOES NOT map to UNFASTENED.
