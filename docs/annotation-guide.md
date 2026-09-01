# Annotation guide

- Dataset scope: label canonical positives only for people visibly occupying a vehicle. A pedestrian or person beside a vehicle is a hard negative, even if holding a phone or not wearing a belt.
- Every sample records a `vehicle_context_id` or source vehicle/clip group. For a pre-cropped cabin image, the whole image is the vehicle context. Assign an occupant role to the sample focus and to every canonical box; rear roles use `rear_left`, `rear_center`, or `rear_right` to match runtime configuration.
- `phone`: tight box around the visible physical phone. Do not include the hand or whole person.
- `seatbelt_fastened`: person upper-body region with clear, correctly routed belt evidence.
- `seatbelt_unfastened`: comparable person upper-body region with positive evidence of no belt.
- Occlusion, glare, darkness, crop, or an invisible belt is `UNCERTAIN`, never automatically unfastened.
- Reject thin belt-object boxes and mounted-phone hard negatives when they do not match canonical semantics.
- A passenger using a phone is allowed at event level, but the physical phone may still be annotated so the detector learns the object; occupant-role association suppresses the event.
- An unfastened passenger is a violation when the upper-body evidence is visible and the passenger is associated with the vehicle.

All imported seatbelt labels begin `PENDING`. Model-assisted review may promote a technically valid proposal
to `MODEL_ASSISTED_PROPOSAL_PENDING_APPROVAL` for explicitly exploratory training, but it must retain
`reviewer_type: AUTOMATED`, model identity, confidence, evidence, and
`governance_eligible: false`.
Only a real, named human decision with the required context metadata may create per-sample
`HUMAN_APPROVED` state and enter the governed dataset.

The reviewer UI shows the model-assisted proposal and evidence before the human form. A batch
acknowledgement may record a visible set of high-confidence proposals, but it is stored as
`ADMIN_ACKNOWLEDGED_MODEL_PROPOSAL_BATCH`; it is not equivalent to reviewing every sample and
never sets governed readiness by itself.
