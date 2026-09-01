# Architecture

Video enters the FastAPI upload service, receives a content hash, and creates an analysis job. V1 uses one process-wide YOLO11s detector. V2 can load separate phone and occupant-upper-body detectors, a three-state seatbelt classifier, a vehicle detector, a pose estimator, and a calibrated logistic fusion artifact. Every model is loaded lazily once per process.

For raw traffic scenes, the vehicle detector and tracker produce vehicle/cabin ROIs before any safety inference. For cabin inputs, camera-calibrated occupant regions associate observations with driver or passenger roles. Phone-to-hand/face geometry, classifier agreement, online sequence features, and temporal persistence produce PHONE or NO_SEATBELT candidates. ByteTrack IDs, vehicle IDs, occupant roles, cooldown, immutable evidence, and human review prevent single-frame detections from becoming repeated automatic conclusions.

The phone detector observes an object, not use. Mounted/static phones, passenger phones, and people outside a vehicle never become PHONE violations. The belt pipeline first detects a comparable upper-body ROI and then classifies `FASTENED`, `UNFASTENED`, or `UNCERTAIN_OR_OCCLUDED`. Low confidence, a small class margin, occlusion, or model disagreement fails closed; invisible belt evidence is never equivalent to unfastened.
