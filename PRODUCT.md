# Product Specification — Roadwatch Driver Safety

<!-- impeccable:product-schema 1 -->

## Platform
Web (Operations Dashboard & Human Review Interface)

## Technology Stack
React with TypeScript, Vite, IBM Carbon Design System, FastAPI, SQLAlchemy (SQLite / MySQL), Ultralytics YOLO11s, ByteTrack, and OpenCV.

## Users
- **Safety Reviewers:** Inspect temporal event evidence packages and confirm, reject, or request review for detected violations.
- **Operations Analysts:** Upload vehicle cabin recordings, monitor processing jobs, search confirmed events, and export audit trails.
- **ML Engineers:** Audit model weights, manage subject-disjoint splits, calibrate temporal parameters, and verify cryptographic locks.

## Product Purpose
Detect visible handheld phone use and unfastened seatbelt states in vehicle cabin camera feeds through a multi-stage decoupled architecture. The system applies vehicle tracking, cabin localization, occupant-role association, pose-guided hand/face proximity context, and temporal hysteresis to generate reviewable `PHONE` and `NO_SEATBELT` events.

## Positioning & Architecture
Rather than a naive monolithic detector, Roadwatch employs specialized decoupled models:
1. **Phone Detector:** YOLO11s (image size 960) localizing physical handheld devices.
2. **Seatbelt Detector:** YOLO11s (image size 1280) localizing occupant upper-body torso ROIs.
3. **Seatbelt Classifier:** YOLO11s-cls evaluating three discrete states (`seatbelt_fastened`, `seatbelt_unfastened`, `uncertain_or_occluded`).
4. **Pose & Context Reasoning:** YOLO11n-pose evaluating hand and facial proximity to distinguish handheld phone use from dashboard mounts.
5. **Temporal Event Engine:** Stateful sliding window with parameter-controlled activation and release thresholds.

## Capabilities and Operational Constraints
- **Phone Violations:** Evaluated strictly for the resolved `driver` role. Passenger phone interactions never trigger driver violations.
- **Seatbelt Violations:** Evaluated across all configured occupant ROIs. `uncertain_or_occluded` states route to `NEEDS_REVIEW` and never default to unfastened violations.
- **Temporal Persistence:** Violations require multi-frame confirmation with defined positive ratio and cooldown periods.
- **Evidence Bundles:** All candidate triggers generate immutable evidence frames, bounding box crops, and metadata JSON traces.

## Verified Evidence on Hand
- **Component Weights:** Verified baseline `V2_BASELINE_001` (`models/locked/v2_baseline_001/model_lock.json`).
- **Validation Metrics:** Phone mAP50 94.48%, Seatbelt Det mAP50 94.73%, Seatbelt Cls Top-1 80.97%.
- **Frozen Test:** Phone mAP50 90.50%, Seatbelt Det mAP50 92.70%, Seatbelt Cls Top-1 78.90% (`FROZEN_TEST_RUN_COUNT = 1`).
- **Temporal Benchmark:** Evaluated on external Vicomtech DMD driver monitoring continuous sequences (`reports/DMD_PHONE_TEMPORAL_BENCHMARK.json`).
- **Runtime Performance:** Measured on local CPU execution (`reports/V2_RUNTIME_BENCHMARK_FINAL.md`).

## Product Principles
- Prefer traceable, tamper-evident evidence over confident heuristic guesses.
- Fail closed on uncertainty: ambiguous context routes to human review.
- Strictly separate component detection metrics, temporal event metrics, and runtime benchmarks.
- Transparently state academic prototype completion without falsely claiming production certification.
