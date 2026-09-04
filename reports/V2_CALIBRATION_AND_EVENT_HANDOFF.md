# V2 Calibration and Event Handoff

## 1. Overview
This handoff report captures the strict transition from raw evaluated capabilities to a `CALIBRATED_EVALUATION_CANDIDATE`. Thresholds were rigorously derived using canonical validation splits to prevent overfitting against the frozen test sets.

- **Status Transition:** `EVALUATION_COMPLETE` -> `CALIBRATED_EVALUATION_CANDIDATE`
- **Frozen Test Consumed:** TRUE (Count = 1)
- **Human Verified:** FALSE
- **Production Ready:** FALSE

## 2. Derived Threshold Locks
By analyzing precision-recall tradeoffs on the validation set, the following thresholds were locked into `models/locked/v2_baseline_001/calibration_lock_v2.json` (`5e88b5040ad9f8b579f1dcfd0232db309e1dccdbf93ea93206790f7ed8f4683e`):

- **Phone Detector (Conf = 0.5996):** Maximizes F1 while ensuring a high precision (96.67%) to filter ambient noise before it hits the temporal logic layer.
- **Seatbelt Detector (Conf = 0.4074):** Maximizes F1 while retaining strong recall (87.12%) to reliably feed the classifier.
- **Seatbelt Classifier:** Implements an asymmetric reject policy where Fastened requires `>= 0.50` and Unfastened requires `>= 0.60`. Anything below is forced to `uncertain_or_occluded` to reduce false violation triggers.

## 3. Pending Scientific Phases
Before the system can be deployed, the following gates remain mathematically unresolved due to missing data:

1. **Temporal Calibration (`PENDING_SEQUENCE_GROUND_TRUTH`):**
   No contiguous video sequences with annotated event boundaries exist to derive temporal smoothing parameters like `min_positive_frames` or EMA values.
2. **Event Evaluation (`PENDING_NEW_UNTOUCHED_HOLDOUT`):**
   The canonical frozen test is officially consumed. End-to-end evaluation metrics MUST be generated using a fresh, completely untouched sequence dataset.

Until these datasets are injected and evaluated, the pipeline is parked in its calibrated state.
