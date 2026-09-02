# V2 Production Readiness

Overall status: **BLOCKED**. The repository now has stronger production boundaries, but data,
weights, calibration, external testing, and human approval remain mandatory.

| Capability | Status | Evidence / blocker |
|---|---|---|
| Immutable V1 baseline | IMPLEMENTED | `experiments/MC_BOOTSTRAP_001/config.yaml` unchanged |
| Vehicle class retained | IMPLEMENTED | car/truck/bus/motorcycle stored in runtime trace |
| Motorcycle seatbelt prohibition | IMPLEMENTED | policy and unit test |
| Explicit cabin localization | IMPLEMENTED | custom/calibrated/geometry strategies and unknown gate |
| Validated windshield weights | BLOCKED_BY_WEIGHTS | no locked windshield artifact |
| Occupant role with unknown/handedness/rear support | IMPLEMENTED | rule-based, confidence-bearing association |
| Production occupant association accuracy | NOT_RUN | requires labeled camera-specific association set |
| Phone-use temporal policy | PARTIALLY_IMPLEMENTED | rule-based; learned action model blocked by data |
| Three-state belt fail-closed policy | IMPLEMENTED | uncertainty, confidence, margin, conflict gates |
| Seatbelt hard negatives | IMPLEMENTED_TOOLING / BLOCKED_BY_DATA | proposal-only queue enforces hashes, group diversity, scenario coverage, and human PENDING state; governed captures missing |
| Human review audit trail | IMPLEMENTED_TOOLING / BLOCKED_BY_HUMAN_REVIEW | schema v2 appends transitions, reason, source hash, reviewer provenance and box identity; no governed decisions exist |
| Phone negative review | IMPLEMENTED_TOOLING / BLOCKED_BY_HUMAN_REVIEW | exact 3,013-row priority queue; 679 priority-1; zero human approvals |
| Data diversity | AUDITED / NOT_GOVERNED | one provider per detector proposal dataset; camera/video/vehicle/person metadata absent; subject isolation NOT_PROVABLE |
| Per-vehicle behavior tracking | IMPLEMENTED | IoU + center + class + continuity |
| EMA/ratio/duration/gap/hysteresis/cooldown | IMPLEMENTED | activation latch, release threshold, expiry/reset and tests |
| Calibrated fusion | BLOCKED_BY_WEIGHTS | artifact absent; health exposes fallback/fail-closed |
| Evidence clip and trace | IMPLEMENTED | canonical keyframes, protected clip/trace, full-package SHA-256 anchor and confirmation gate |
| Human review provenance | IMPLEMENTED | explicit HUMAN assertion, append-only transition/source hashes, idempotent identical retry and serialized local append; human identity remains an operational responsibility |
| Review 1 delegated review | PARTIAL / NOT_GOVERNED | append-only AI provenance, SHA validation, checkpoints and separate materialization lanes implemented; 200/3,013 phone-negative candidates visually reviewed, 2,950 attention items remain |
| Admin confirmation | IMPLEMENTED_TOOLING / NOT_RUN | explicit typed confirmation appends a separate HUMAN record only after metadata/role/condition gates; zero actual confirmations |
| Analysis queue/concurrency | PARTIALLY_IMPLEMENTED | API/business logic separated; single-process demo worker serializes complete inference; BackgroundTasks is not durable or multi-process |
| File source | IMPLEMENTED | `FileVideoSource` |
| Webcam/RTSP production source | PARTIALLY_IMPLEMENTED | malformed identifiers and failed-open cleanup tested; reconnect/read-timeout/load evidence absent, so adapters remain experimental |
| Event evaluator | IMPLEMENTED_TOOLING | requires frozen reviewed truth + ACTIVE governed model lock; binds input hashes, refuses result overwrite and can re-verify every bound artifact; measurements `NOT_RUN` |
| Association/pose evaluator | IMPLEMENTED_TOOLING | measurements `NOT_RUN` |
| Runtime benchmark | IMPLEMENTED_TOOLING | measurements `NOT_RUN` |
| Governed V2 training | BLOCKED_BY_HUMAN_REVIEW | readiness report gates remain open |
| External frozen test | IMPLEMENTED_TOOLING / BLOCKED_BY_DATA | freeze tool enforces reviewed source/camera/video-disjoint inputs and immutable output; real set absent |
| Calibration/model-lock binding | IMPLEMENTED_TOOLING / NOT_RUN | lock verifies calibration model SHA, exact configured thresholds and all configured component hashes; real calibration absent |
| Production startup validation | IMPLEMENTED | refuses mismatched/unlocked/unapproved artifacts, missing configured components and the public development secret |
| Production activation | BLOCKED | no ACTIVE human-approved matching model lock; all scientific gates above must pass |
| Deterministic CI | IMPLEMENTED | Ruff, compile, tests, FastAPI import and frontend build; no training/model/data downloads |
| Main branch protection | BLOCKED_BY_GITHUB_SETTINGS | public branch metadata reports `protected=false`; see `docs/BRANCH_PROTECTION_RECOMMENDATION.md` |

## Required activation record

`training.lock_model` records experiment id, weight/config/data hashes, validation artifact,
calibration artifact, creation time, activation state, environment, and explicit
`production_approved: false`. A file named `best.pt` is never sufficient for activation.

## Priority order

1. Finish human semantic review and source grouping without auto-approval.
2. Capture/review hard negatives and independent uncertain/occluded belt examples.
3. Freeze a source/camera-disjoint external test set before model selection.
4. Train V2 components independently; retain V1 as the baseline.
5. Evaluate components and association on validation; calibrate thresholds/fusion on validation.
6. Lock model/version artifacts and run the frozen test once.
7. Build human event ground truth and run event-level evaluation.
8. Validate camera cabin/seat geometry, run target-hardware benchmark, then add a durable queue.

The external set can only be frozen with `training/freeze_external_test.py`. Its manifest must
point to real video and annotation files, carry reviewer provenance, meet
`datasets/v2_external_test_policy.yaml`, and remain disjoint from the supplied development
manifest. The generated lock asserts integrity and isolation only; it never asserts accuracy.
`training.freeze_event_ground_truth` then binds reviewed event truth to that external lock.
`training.evaluate_events` will only measure it with an ACTIVE governed V2 model lock and will not
overwrite the first result.
