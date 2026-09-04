# V2 Temporal Calibration Plan

## Status: PENDING_SEQUENCE_GROUND_TRUTH

This document outlines the evaluation architecture for smoothing single-frame inferences into continuous semantic events. It cannot be executed or tuned until sequence-level ground truth (GT) is available.

## 1. Parameters to Tune (Post-GT)
Once sequence GT is obtained, the following hyperparameters will be swept and locked to maximize event-level F1:
- `observation_window`
- `minimum_observations`
- `minimum_positive_duration`
- `positive_ratio`
- `ema_alpha`
- `candidate_threshold`
- `activation_threshold`
- `release_threshold`
- `gap_tolerance`
- `cooldown`

## 2. Phone Pipeline Event Logic
**Progression:**
`vehicle -> cabin -> occupant association -> DRIVER -> phone detection/association -> hand/face/ear/pose context -> handheld-use evidence -> temporal confirmation -> PHONE_USE event`

**Reject Conditions (Safety Invariants):**
- `passenger_phone`: Passenger phone use MUST NOT trigger `PHONE_USE`.
- `mounted_static_phone`: Phones mounted to the dash/windshield MUST NOT trigger `PHONE_USE`.
- `outside_person`: Persons outside the vehicle MUST NOT trigger any violation.
- `unknown_occupant_role`: If the occupant role cannot be determined, it MUST NOT default to a violation.
- `insufficient_context`: Missing vehicle/cabin context MUST fail-closed.
- `single_frame_spike`: An isolated frame of evidence MUST NOT create an event.

## 3. Seatbelt Pipeline Event Logic
**Progression:**
`vehicle -> cabin -> occupant -> upper_body_detection -> 3_class_classifier -> reject_UNKNOWN_policy -> temporal confirmation -> NO_SEATBELT event`

**Reject Conditions (Safety Invariants):**
- `uncertain_or_occluded`: Explicitly rejected by the static classifier threshold policy.
- `missing_visibility`: If the upper body is not visible, fail-closed.
- `motorcycle`: Motorcycles MUST NOT trigger a seatbelt violation.
- `outside_person`: Persons outside the vehicle MUST NOT trigger any violation.
- `unknown_context`: Missing vehicle/cabin context MUST fail-closed.
- `single_frame_spike`: An isolated frame of evidence MUST NOT create an event.
