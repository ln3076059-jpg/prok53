# Senior AI / Computer Vision Audit

Audit date: 2026-09-02
Repository state reviewed: current working tree based on `main`
Scientific baseline protected: `experiments/MC_BOOTSTRAP_001/config.yaml`

## Executive conclusion

The repository has a strong reproducibility and dataset-governance foundation, but the V2
runtime is not production-ready. The blocking issues are ground-truth approval, independent
external test data, trained and calibrated V2 artifacts, real cabin localization, reliable
occupant association, and event-level evaluation. No metric, approval, or readiness claim is
inferred from code structure.

V1 `MC_BOOTSTRAP_001` remains an immutable scientific baseline. It intentionally preserves the
single three-class detector so the thesis can measure the partial-label conflict: phone-oriented
images may contain unlabeled seatbelts and seatbelt-oriented images may contain unlabeled phones;
ordinary YOLO training can therefore learn valid but unlabeled objects as background. V2 is a
separate architecture designed to remove that cross-task label conflict.

## Findings

| Severity | Finding | Root cause | Consequence | Related files | Remediation | Status after this upgrade |
|---|---|---|---|---|---|---|
| CRITICAL | Ground truth is not governed-ready | 9,728 phone and 4,868 seatbelt proposals remain pending; group and diversity metadata are incomplete | Training/evaluation claims would not be defensible | `reports/v2_training_readiness.md`, `datasets/manifests/` | Preserve proposal lane, require append-only human approval and group-clean capture | BLOCKED_BY_HUMAN_REVIEW |
| CRITICAL | Subject and camera isolation cannot be proven | Proposal manifests omit camera, video, vehicle and person identities | Subject-disjoint and cross-camera generalization claims are unsupported | `reports/V2_DATA_DIVERSITY_AUDIT.json` | Capture trustworthy identities and keep each identity in one development split | AUDITED; NOT_PROVABLE |
| CRITICAL | No final V2 model or calibrated thresholds | Runtime version is `MULTIMODEL_V2_UNTRAINED`; thresholds are placeholders | Runtime confidence is not a validated operating point | `models/model_config_v2.yaml`, `models/active/v2/` | Fail closed; require locked hashes and validation calibration before activation | BLOCKED_BY_WEIGHTS |
| CRITICAL | Whole vehicle bbox is treated as cabin | No windshield/cabin localization stage exists | Occupants and small objects lose pixels and background causes false positives | `backend/services/video_analyzer.py`, `backend/ai/auxiliary.py` | Add explicit `CabinLocalizer`; never silently relabel a vehicle crop as cabin | IMPLEMENTED |
| CRITICAL | V2 temporal config is incompatible with the event engine constructor | Runtime passes smoothing keys to an engine that does not accept them | V2 analysis can fail before frame inference | `backend/services/video_analyzer.py`, `backend/ai/events.py`, `models/model_config_v2.yaml` | Use an explicit temporal configuration contract and validate it | IMPLEMENTED |
| HIGH | Vehicle type is discarded | `VehicleRegion` omits COCO class id/name | Motorcycle can incorrectly enter seatbelt policy | `backend/ai/auxiliary.py` | Preserve class and prohibit motorcycle seatbelt events | IMPLEMENTED |
| HIGH | Occupant role is fixed-ROI-only | Association returns the largest overlap with static regions | Camera angle, handedness, and seat layout changes cause role errors | `backend/ai/association.py` | Add geometry, handedness, confidence, calibrated fallback and unknown role | IMPLEMENTED |
| HIGH | Behavior identities use a pseudo ID | Missing IDs are synthesized from vehicle id and class id | Different people/phones of the same class collapse into one temporal track | `backend/services/video_analyzer.py` | Add per-vehicle track-by-detection using IoU, center distance, class and continuity | IMPLEMENTED |
| HIGH | Phone detection is too close to phone-use semantics | Context uses only sparse keypoint-to-phone geometry and old labels | Mounted phones and ambiguous phone observations can become unstable | `backend/ai/auxiliary.py`, `backend/ai/events.py` | Normalize by occupant scale, retain explicit context states, require persistence | IMPLEMENTED_WITH_RULE_BASED_LIMITATION |
| HIGH | Seatbelt hard negatives are missing | Every current ROI detector image contains an upper-body box | Empty seats, reflections and straps can trigger false positives | `reports/v2_training_readiness.md` | Use the governed proposal-only hard-negative queue, then independently capture and human-review every required scenario | IMPLEMENTED_TOOLING; BLOCKED_BY_DATA |
| HIGH | Zero-box phone proposals are unsafe negatives | A missing source box can hide an unlabeled phone | Visible phones may be trained as background | `datasets/manifests/v2_phone_negative_priority_review.json` | Review model detections, uncertainty, adverse visibility and phone-like confusers first | IMPLEMENTED_TOOLING; BLOCKED_BY_HUMAN_REVIEW |
| HIGH | Uncertain belt ground truth is not governed-ready | Model-assisted uncertainty is exploratory and pending approval | A three-state production classifier cannot be claimed | `docs/v2-training.md`, pending manifests | Keep uncertainty distinct and fail closed on low confidence/margin | BLOCKED_BY_HUMAN_REVIEW |
| HIGH | No source-disjoint external test | Current splits share provider/domain | Generalization to traffic cameras is unknown | `reports/v2_training_readiness.md` | Freeze a new camera/provider test set before final model selection | BLOCKED_BY_DATA |
| MEDIUM | Temporal logic lacked an activation latch | Cooldown alone did not use the configured release threshold | A continuous violation could reactivate after cooldown without negative evidence | `backend/ai/events.py`, `backend/ai/sequence.py` | Add event-scoped hysteresis latch, explicit release, expiry and vehicle reset | IMPLEMENTED |
| MEDIUM | Fusion mode is implicit | Missing artifact silently selects a rule fallback | Operators cannot tell calibrated from fallback output | `backend/ai/fusion.py`, `/health` | Expose mode, availability, artifact hash and threshold; support calibrated-only gate | IMPLEMENTED |
| MEDIUM | Evidence lacked a confirmation-grade integrity gate | Persistence checked a DB row, not every required file/hash or post-event completeness | A reviewer could confirm incomplete or altered evidence | `backend/ai/evidence.py`, `backend/api/routes.py`, `frontend/src/App.tsx` | Require complete clip, canonical keyframes, trace status, and matching SHA-256 before CONFIRMED | IMPLEMENTED |
| MEDIUM | API execution is coupled to FastAPI background tasks | Route function owns worker business logic | No independent scheduling/retry boundary | `backend/api/routes.py` | Add queue/worker abstraction while retaining demo adapter | PARTIALLY_IMPLEMENTED |
| MEDIUM | Event-level metrics are absent | No frozen human event ground truth | Detector AP cannot establish violation accuracy | `training/evaluate_v2.py`, reports | Add event evaluation contract/tool; leave results `NOT_RUN` | IMPLEMENTED_TOOLING; NOT_RUN |
| HIGH | Event evaluator did not enforce its frozen-test claim | Arbitrary CSVs could be evaluated without checking the external truth or model lock, and results could be overwritten | A mutable or pre-lock run could be mistaken for final external evidence | `training/evaluate_events.py`, `training/freeze_external_test.py` | Bind reviewed truth/external/model hashes, require ACTIVE governed lock and refuse overwrite | IMPLEMENTED_TOOLING; NOT_RUN |
| MEDIUM | UI does not serve real evidence or history | Event detail contains a deployment placeholder | Human review is not evidence-backed in the app | `frontend/src/App.tsx`, `backend/api/routes.py` | Serve protected evidence and append-only review provenance | IMPLEMENTED |
| LOW | Canonical config env name is inconsistent | `.env.example` uses `MODEL_CONFIG` instead of the settings field name | Deployments can silently load V1/default config | `.env.example`, docs | Standardize on `MODEL_CONFIG_PATH` and validate startup/runtime status | IMPLEMENTED |
| HIGH | Production startup trusted configuration too loosely | Artifact existence was the primary route gate | Mismatched class maps, uncalibrated thresholds, absent fusion, or an unlocked model could start | `backend/ai/detector.py`, `backend/main.py` | Validate config contract in every environment and model-lock/approval/calibration hashes in production | IMPLEMENTED |
| MEDIUM | Benchmark omitted cabin and fusion stages | Component timing did not follow the configured input scope | Report could not characterize the full V2 chain | `tools/benchmark_runtime.py` | Measure decode, vehicle, cabin, behavior, pose, classifier, fusion, total p50/p95 and memory | IMPLEMENTED_TOOLING; NOT_RUN |
| MEDIUM | Review decisions lacked explicit transitions and reason codes | Append-only lines recorded status but not the previous/new state contract | Revisions were harder to reconstruct and audit | `tools/annotation_reviewer/app.py` | Record schema-v2 transition, reason, source hash, stable box id and decision id | IMPLEMENTED |
| MEDIUM | Threshold artifact lacked model/dataset provenance | Calibration output contained thresholds but not binding hashes | A threshold could be applied to the wrong model or validation data | `training/calibrate_thresholds.py` | Require model and validation manifest; record their SHA-256 and validation-only method | IMPLEMENTED_TOOLING; NOT_RUN |

## Production gates that remain closed

- Governed V2 training: **BLOCKED_BY_HUMAN_REVIEW**.
- V2 weights: **UNTRAINED / BLOCKED_BY_WEIGHTS**.
- Threshold and fusion calibration: **NOT_RUN**.
- Frozen external-domain test: **NOT_RUN / BLOCKED_BY_DATA**.
- Event-level evaluation: **NOT_RUN / BLOCKED_BY_DATA**.
- Production camera calibration and load benchmark: **NOT_RUN**.

The code improvements below create enforceable boundaries and auditability. They do not clear
any scientific, human-approval, data, model, or production gate.

## Verification snapshot

Verification performed on 2026-09-02 against the working tree described above:

| Check | Result |
|---|---|
| V1 scientific baseline diff | PASS — `experiments/MC_BOOTSTRAP_001/config.yaml` unchanged |
| Python test suite | PASS — 106/106 tests locally |
| Ruff checks and Python compilation | PASS |
| Frontend production build | PASS |
| FastAPI in-memory startup and `/health` smoke test | PASS — HTTP 200, deliberately `degraded` without approved V2 artifacts |
| Production fail-closed startup contract | PASS — refused missing/unapproved/uncalibrated artifacts |
| Review UI mechanical quality detector | COMPLETED — flagged one color-token advisory and type-hierarchy warning; both corrected, detector not rerun per bounded-pass policy |
| CI workflow | PASS — Ubuntu GitHub Actions run `33586606094` verified fatal Ruff, compile, Python tests, FastAPI import and frontend build for commit `f95c6c8` |
| Runtime benchmark | NOT_RUN — no production video/model artifacts supplied |
| Association and event-level scientific evaluation | NOT_RUN — no frozen human ground truth supplied |

These checks validate code behavior and refusal paths only. They are not evidence of model
accuracy, external-domain generalization, production approval, or deployment readiness.
