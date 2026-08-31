# Architecture

Video enters the FastAPI upload service, receives a content hash, and creates an analysis job. One process-wide `SafetyDetector` loads the locked YOLO11s weights. ByteTrack maintains IDs. Driver-region overlap and configurable temporal persistence produce PHONE or NO_SEATBELT candidates. Events, detections, immutable evidence paths, and human reviews are stored in MySQL and exposed to the React dashboard.

The detector observes a phone object, not phone use. Unfastened status requires a positive model observation, valid driver association, and persistence. Conflicting seatbelt states become `NEEDS_REVIEW`.

