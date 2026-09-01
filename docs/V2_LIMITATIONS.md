# V2 Limitations

## Current scientific status

V2 is **UNTRAINED**, **NOT CALIBRATED**, **NOT HUMAN APPROVED**, and **NOT PRODUCTION READY**.
No component or event metric has been run for a final governed V2 model.

## Data blockers

- 9,728 phone proposals and 4,868 seatbelt proposals remain pending human semantic review.
- Source grouping, vehicle/person/camera identities, adverse-condition coverage, and independent
  external domains are incomplete.
- Phone negatives are not human-confirmed.
- Seatbelt ROI data lacks reviewed empty cabin/seat, reflection, steering wheel, clothing fold,
  bag/strap, non-occupant, and partial-occupant hard negatives.
- The hard-negative queue builder validates proposals and preserves human-review gates, but it
  cannot supply the missing independent captures or approve them.
- The exact hard-negative capture manifest is absent: all 10 required seatbelt scenarios have zero
  captured review candidates and remain `BLOCKED_BY_DATA`.
- The diversity audit found one provider in each detector proposal dataset and no declared camera,
  video, vehicle, or person identities. Subject-disjoint status is `NOT_PROVABLE`.
- The prioritized 3,013-row phone-negative queue is model-assisted triage only; all rows remain
  `PENDING` and human-approved count is zero.
- Governed uncertain/occluded examples are missing from one or more classifier splits.
- Model-assisted pending-approval data remains exploratory and cannot be promoted automatically.

## Runtime limitations

- No trained windshield detector is supplied. Geometry cabin ROIs are proposals and default below
  the acceptance threshold; traffic-scene inference therefore fails closed without approved
  calibration or custom weights.
- Occupant association is rule-based. It exposes confidence and `unknown`, but is not a validated
  seat-position model and has no identity ReID.
- Generic COCO pose has a large domain gap through distant, reflective, dark, or occluded glass.
- Phone-use inference is rule-based temporal evidence, not trained action recognition. Gaze and
  optical flow are not implemented.
- Local track-by-detection handles short gaps but does not solve long occlusion or cross-camera
  identity.
- Evidence MP4 generation depends on the local OpenCV codec. A missing codec records
  `CLIP_UNAVAILABLE`; it never fabricates a clip and confirmation remains blocked.
- BackgroundTasks remains a non-durable demo queue. Process crashes can lose queued work.
- Webcam/RTSP adapters are experimental and lack production reconnect/backpressure controls.
- Database schema migration tooling is still required before production deployment.
- Production startup validation is implemented, but no ACTIVE human-approved V2 model lock exists;
  production therefore correctly refuses to start with the current artifacts.

## Evaluation limitations

- Component metrics: `NOT_RUN` for final V2 weights.
- Association and pose-availability metrics: `NOT_RUN`; tooling exists, but no independent
  association ground truth exists.
- Event metrics: `NOT_RUN`; no frozen human event ground truth exists.
- External-domain test: `NOT_RUN` and not yet frozen.
- Runtime benchmark: `NOT_RUN` until `py -m tools.benchmark_runtime --video ...` runs on the target
  hardware with installed weights.

Changing YOLO size or increasing epochs does not clear any limitation above.
