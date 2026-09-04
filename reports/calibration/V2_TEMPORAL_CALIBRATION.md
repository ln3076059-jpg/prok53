# V2 Temporal Calibration Status

**Status:** `PENDING_SEQUENCE_GROUND_TRUTH`

## Overview
Temporal calibration requires adjusting parameters such as `observation_window_frames`, `min_positive_frames`, `EMA alpha`, `max_gap_frames`, and event thresholds to prevent single-frame flickering from causing false violation alarms.

## Blocker
Currently, the repository does not contain a validated sequence dataset (e.g., continuous video clips annotated with exact violation start and end times). 
Without sequence ground truth, any temporal parameters would be purely fabricated/heuristic rather than data-driven, which violates the strict governance rules.

## Next Steps
To unlock the Temporal Policy Lock, a new dataset of annotated continuous sequences (not just still frames) must be introduced into the canonical dataset registry. Only then can the temporal smoothing policy be mathematically verified and locked.
