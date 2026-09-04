# V2 Temporal Data Readiness

## Status: PENDING_SEQUENCE_GROUND_TRUTH

The model pipeline currently operates strictly on isolated frames and boxes. System-level evaluation and event generation require temporal smoothing and continuity tracking over time. Before temporal hyperparameters can be calibrated (e.g., EMA alpha, confirmation duration, cooldown), we must have sequence-level ground truth.

### Current Data Availability
- **Sequence Count:** 0
- **Clip Identity / Frame Timestamps:** Missing
- **Phone Event Intervals:** Missing
- **Seatbelt Event Intervals:** Missing
- **Occupant Roles & Vehicle Context:** Missing
- **Human Review:** Missing

### Blockers for Temporal Calibration
Temporal calibration CANNOT proceed until the following blockers are cleared:
1. Availability of un-sampled, continuous video clips or frame sequences.
2. Independently reviewed sequence-level ground truth (GT) explicitly demarcating the start and end of `PHONE_USE` and `NO_SEATBELT` events.
3. GT including context annotations (e.g., occupant roles, vehicle context) to properly evaluate filtering rules.

**Conclusion:** The repository is `PENDING_SEQUENCE_GROUND_TRUTH`. Temporal policy locks and end-to-end event evaluation are completely blocked.
