# ROADWATCH V20 — Actual Autonomous Pipeline Call Graph
```
RAW VIDEO STREAM (MP4 / Camera Ingestion)
  │ [cv2.VideoCapture]
  ▼
DRIVER ROI LOCALIZATION [Cabin Geometry / Cabin Localizer]
  │ Crop Driver Cabin Region at Native Resolution (832x612 on 1280x720)
  ▼
NATIVE-RESOLUTION PHONE DETECTION [AutonomousVideoInferenceEngine]
  │ Ensemble of Specialist YOLO11s (conf>=0.15) + Auxiliary YOLO11n (class 67, conf>=0.15)
  │ Bounding box merging via confidence-ordered NMS (IoU >= 0.50)
  │ Global frame coordinate translation
  ▼
SPATIAL KEYPOINT CONTEXT EVALUATION [PoseEstimator + classify_phone_context]
  │ Evaluates driver hand and face proximity to detected phone box
  │ HANDHELD_PHONE_USE: hand proximity > 0.35 OR (face > 0.35 and hand > 0.20)
  │ MOUNTED_OR_STATIC: distance to hands/face > threshold
  │ UNKNOWN: pose unavailable or ambiguous
  ▼
DRIVER ROLE ASSOCIATION [OccupantAssociator]
  │ Ensures detection belongs to verified driver track
  ▼
TEMPORAL EVENT ENGINE [TemporalEventEngine]
  │ Parameters: window=2.0s, min_positive=0.5s, activation=0.25, bridge=4.0s, cooldown=3.0s
  │ UNKNOWN observations preserve hysteresis without resetting engine
  ▼
EVENT CANDIDATE EMISSION
  │ Dedicated PHONE events with start/end timestamps
  ▼
POST-HOC FROZEN MATCHER [match_temporal_events]
  │ Matching against OpenLABEL ground truth ONLY after video processing finishes
```
