# V2 Multi-Stage Architecture

V2 is independent from the immutable `MC_BOOTSTRAP_001` V1 baseline. It separates physical
phone detection from phone-use evidence and separates upper-body localization from three-state
seatbelt classification.

```text
raw frame
  -> vehicle detection + ByteTrack (id, class, confidence, bbox)
  -> CabinLocalizer
       custom windshield model
       OR approved per-camera ROI
       OR low-confidence geometry proposal
  -> UNKNOWN_CABIN gate
  -> behavior detectors in rectified cabin ROI
  -> per-vehicle IoU/center-distance track-by-detection
  -> occupant association (geometry + handedness + calibration + confidence)
       driver | front_passenger | rear_left | rear_center | rear_right | unknown
  -> phone context / three-state belt classifier
  -> explicit CALIBRATED_MODEL or RULE_FALLBACK fusion
  -> EMA + positive ratio + duration + gap tolerance + hysteresis + cooldown
  -> event policy
  -> pre/post evidence clip + key frames + JSON trace + SHA-256
  -> append-only human review
```

## Fail-closed gates

- A raw vehicle crop is never silently called a cabin. Cabin confidence below the configured
  threshold produces `UNKNOWN_CABIN` and behavior inference is skipped.
- An ambiguous seat position is `unknown`, never driver by default.
- Passenger phone observations produce no event. Persistent unknown-role or unknown-context
  phone evidence produces `NEEDS_REVIEW`.
- `MOUNTED_OR_STATIC` phone context produces no event.
- `uncertain_or_occluded`, low classifier confidence, or low top-1 margin cannot produce an
  automatic `NO_SEATBELT` candidate.
- A motorcycle can never produce `NO_SEATBELT`.
- Missing required calibrated fusion reports a fail-closed result and remains visible on
  `/health`.

## Runtime components

| Component | Implementation | Replacement boundary |
|---|---|---|
| Vehicle tracking/type | COCO vehicle detector with ByteTrack; class retained in `VehicleRegion` | Custom road-traffic detector |
| Cabin localization | `backend/ai/cabin.py` | Custom windshield YOLO or rectification model |
| Local object tracking | `backend/ai/tracking.py` | ReID/DeepSORT/learned association |
| Occupant association | `backend/ai/association.py` | Seat/occupant detector plus camera calibration |
| Phone evidence | physical phone + scale-normalized pose geometry + persistence | TCN/LSTM/Transformer/action recognizer |
| Seatbelt evidence | upper-body detector + 3-state classifier | Better domain-specific classifier |
| Temporal/fusion | explicit feature buffers, event hysteresis, logistic/rule modes | Learned sequence fusion after governed labels |
| Evidence | `EventEvidenceBuffer` | Durable object storage/transcoding service |

## Input and execution abstractions

`VideoSource` has file, webcam, and RTSP adapters. Upload jobs currently activate only
`FileVideoSource`; webcam/RTSP remain experimental because reconnect, backpressure, credential
handling, and SSRF controls are not production-complete.

`InferenceWorker` owns analysis business logic. `BackgroundTaskAnalysisQueue` is the current demo
adapter. A durable Redis/RabbitMQ/Celery worker can replace it without moving inference back into
the HTTP route.

## Evidence contract

Every finalized evidence directory contains, when the codec is available:

- `original.jpg`
- `annotated.jpg`
- `evidence.mp4` with requested pre/post context
- `trace.json` with vehicle, cabin, occupant, phone/pose/belt, fusion, temporal, model, threshold,
  timestamp, and per-file SHA-256 evidence

Evidence endpoints require authentication. Review history is append-only in the `reviews` table.
