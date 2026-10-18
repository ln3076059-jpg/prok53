# ROADWATCH V19 — Full Codebase & Architecture Audit
## 1. Executive Summary & Audit Scope
A comprehensive audit of first-party code was conducted across **287 Python files**:
- `backend/` (29 files): FastAPI routes, inference workers, camera ingestion, cabin localization, safety detectors, occupant association, event engines, database models.
- `training/` (207 files): DMD adapters, dataset governance, split validation, causal motion encoders, online candidates, temporal evaluation, human review queues.
- `tests/` (51 files): Unit, integration, regression, and governance test suites.

## 2. Reconstructed End-to-End Pipeline Call Graph
```
VIDEO INPUT (Camera / File)
  │ [cv2.VideoCapture.read() / grab()]
  ▼
VEHICLE / CABIN LOCALIZATION [CabinLocalizer.localize()]
  │ Geometric Cabin ROI / Cabin YOLO (threshold=0.35)
  ▼
OCCUPANT / DRIVER ROI [CabinLocalizer.extract_driver_crop()]
  │ Bounding box extraction & coordinate normalization
  ▼
PHONE DETECTOR [SafetyDetector.predict()]
  │ models/active/v2/phone_detector.pt (YOLO11s, mAP50=94.48%)
  │ Downscales full frame (1280x720) to 640x640 -> Small phone bottleneck (<20px)
  ▼
SOURCE-RESOLUTION / SAHI [AutonomousPhonePipeline.propose()]
  │ Auxiliary YOLO11n on native-resolution driver crop recovers phone proposals
  ▼
ACTION FEATURE ENCODER [OnlineE3.push()]
  │ CausalMotionEncoder: 8-frame buffer, 64x64 uint8, spatial standardization
  ▼
TRAINED ACTION HEAD [NumericHead.predict()]
  │ JSON-only linear head; decision function argmax, OvR sigmoid probabilities
  ▼
DRIVER-PHONE ASSOCIATION [OccupantAssociator.associate() / classify_phone_context()]
  │ Face/hand proximity, role verification -> HANDHELD_USE, MOUNTED, UNKNOWN
  ▼
FUSION [min(raw_phone_score, action_score)]
  │ Rejection rules: unverified driver association, non-handheld context
  ▼
TEMPORAL EVENT ENGINE [TemporalEventEngine.add()]
  │ models/temporal_config_v2.json: hysteresis, activation_threshold, occlusion_bridge
  ▼
EVENT MATCHING & EVALUATION [match_temporal_events()]
  │ DMD_EVENT_MATCHING_POLICY_FREEZE.json (IoU >= 0.3, onset tol <= 2.5s)
```

## 3. Discovered Vulnerabilities & Architecture Insights
1. **Pre-V18 Engine Reset Bug (E2):** In `online_candidate.py`, any unadmitted observation recreated `TemporalEventEngine`, destroying hysteresis across momentary dropouts.
2. **Context Normalization (E1):** Discrepancy between `HANDHELD_USE` and `HANDHELD_PHONE_USE` caused admitted observations to be rejected.
3. **Downscaling Bottleneck:** 11 of 13 historical FN events failed at the phone detector level because full-frame resize blurred small handheld phones below detector feature stride.
