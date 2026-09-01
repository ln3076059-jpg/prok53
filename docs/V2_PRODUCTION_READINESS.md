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
| Seatbelt hard negatives | BLOCKED_BY_DATA | governed examples missing |
| Per-vehicle behavior tracking | IMPLEMENTED | IoU + center + class + continuity |
| EMA/ratio/duration/gap/cooldown | IMPLEMENTED | temporal engine and tests |
| Calibrated fusion | BLOCKED_BY_WEIGHTS | artifact absent; health exposes fallback/fail-closed |
| Evidence clip and trace | IMPLEMENTED | protected files and SHA-256 |
| Human review provenance | IMPLEMENTED | append-only history exposed to UI |
| Durable analysis queue | PARTIALLY_IMPLEMENTED | abstraction exists; demo adapter is not durable |
| File source | IMPLEMENTED | `FileVideoSource` |
| Webcam/RTSP production source | PARTIALLY_IMPLEMENTED | adapters experimental |
| Event evaluator | IMPLEMENTED_TOOLING | measurements `NOT_RUN` |
| Runtime benchmark | IMPLEMENTED_TOOLING | measurements `NOT_RUN` |
| Governed V2 training | BLOCKED_BY_HUMAN_REVIEW | readiness report gates remain open |
| External frozen test | IMPLEMENTED_TOOLING / BLOCKED_BY_DATA | freeze tool enforces reviewed source/camera/video-disjoint inputs; real set absent |
| Production activation | BLOCKED | all scientific gates above must pass |

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
