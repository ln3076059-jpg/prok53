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
  `CLIP_UNAVAILABLE`; it never fabricates a clip.
- BackgroundTasks remains a non-durable demo queue. Process crashes can lose queued work.
- Webcam/RTSP adapters are experimental and lack production reconnect/backpressure controls.
- Database schema migration tooling is still required before production deployment.

## Evaluation limitations

- Component metrics: `NOT_RUN` for final V2 weights.
- Association metrics: `NOT_RUN`; no independent association ground truth exists.
- Event metrics: `NOT_RUN`; no frozen human event ground truth exists.
- External-domain test: `NOT_RUN` and not yet frozen.
- Runtime benchmark: `NOT_RUN` until `tools/benchmark_runtime.py --video ...` runs on the target
  hardware with installed weights.

Changing YOLO size or increasing epochs does not clear any limitation above.
