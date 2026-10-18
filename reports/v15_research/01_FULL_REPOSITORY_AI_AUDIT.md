# Full repository AI audit

Audit scope: the requested `backend/ai`, `training/dmd`, model configs, registry/API integration, frontend API use, evidence and review paths were read directly. There is no `configs/` directory. The runtime and research stacks share concepts but are separate implementations.

| Stage | Runtime implementation | Input → output / coordinates | Failure and activation state |
|---|---|---|---|
| Decode/context | `backend/services/video_analyzer.py` | uploaded video → BGR frames and vehicle/cabin contexts | rejects unsupported scope; analysis worker serializes inference |
| Detection | `backend/ai/detector.py` | HWC frame/crop → `NormalizedDetection` in input-pixel XYXY | lazy YOLO; thresholded by config; configured weights can be absent; candidate is development-only |
| Driver localization | `backend/ai/cabin.py`, `association.py` | cabin/person boxes → explicit role/confidence | geometry below confidence stays UNKNOWN; tracking never proves role |
| ROI | runtime crop helpers; research `behavior_roi.py`, `high_res_roi.py` | source XYXY → crop plus scale/padding provenance | research source-resolution path is not wired into production `VideoAnalyzer` |
| Pose/phone context | `backend/ai/auxiliary.py` | crop → COCO keypoints/context | missing weights/evidence stays unknown; mounted/static is non-use |
| Action | runtime has no fitted E3; research `online_candidate.py`, `action_model.py`, `causal_motion.py` | causal 64×64 RGB history → 444-D E3 → three scores | fitted head/scaler missing; normalized OvR sigmoid is explicitly uncalibrated |
| Association | `association.py`, `tracking.py` | detections/occupants → role and stable IDs | overlapping/low-confidence roles abstain; IoU tracker differs from configured ByteTrack vehicle path |
| Fusion | `backend/ai/fusion.py`; research `training/dmd/fusion.py` | detector/context/pose/temporal features → score | runtime may use a rule fallback unless calibrated fusion is required; research multiview code is not the same runtime path |
| Temporal event | `backend/ai/sequence.py`, `events.py` | timestamped observations → candidates | driver-only phone gate; strict unknown context becomes NEEDS_REVIEW; cooldown/gap/occlusion rules are config-driven |
| Evidence/review | `evidence.py`, API routes, `research_review.py` | event frames/traces → hashed files and review receipts | file traversal/duplicates rejected; human and AI provenance remain distinct |

The frontend does not run models; it starts analysis through FastAPI, polls job/event APIs, and presents evidence/review state. The active candidate config is `models/model_config_v4_candidate.yaml`: inherited unvalidated thresholds, `production_approved=false`, strict phone context, missing per-camera calibration and no locked independent validation. `model_config.yaml` and AI85 research configs represent different historical/candidate contracts, so their thresholds must not be mixed.

The largest duplication risk is `training/dmd/fusion.py` versus `backend/ai/fusion.py` and research `OnlineE3` versus the production `VideoAnalyzer`. A future model should be exported as one versioned artifact and consumed by the runtime adapter; it should not create a second event engine.
