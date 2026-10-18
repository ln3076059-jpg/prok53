# 13. Independent gA-1 Evaluation Protocol & Findings
- **Protocol:** Evaluated on `gA-1 s2` body camera video (458.1s, 13,633 frames, 8 ground truth phone events).
- **Frozen Models:** `models/active/v2/phone_detector.pt` (SHA-256 `840a29cb...`), `models/auxiliary/yolo11n.pt`.
- **Temporal Configuration:** Calibrated V2 policy (`min_positive_seconds = 0.5s`, `min_observations = 2`, `activation_threshold = 0.25`).
- **Independent AI85 Decision:** `INDEPENDENT_AI85_NOT_ESTABLISHED`.
  - While gA-1 is locked and uncontaminated, establishing a scientifically valid AI85+ claim requires a multi-subject independent test panel (>= 5 drivers) rather than a single subject.
