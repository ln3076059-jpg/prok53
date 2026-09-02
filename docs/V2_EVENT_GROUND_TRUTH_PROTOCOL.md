# V2 Event Ground-Truth Protocol

Status: **BLOCKED_BY_DATA / NOT_RUN**.

## Video set

Use real clips covering normal driving, driver phone use, phone near face, passenger phone,
mounted phone, fastened/unfastened/occluded belt, reflections, low light, vehicle entry/exit and
multiple vehicles. Synthetic-only clips may test logic but cannot establish production accuracy.

## Annotation schema

Each event CSV has `video_id`, `event_id`, `event_type`, `start_seconds`, `end_seconds`,
`vehicle_id`, `cabin_id`, `occupant_role`, `visibility`, `conditions`, `human_review_status`,
`reviewer_id`, `reviewer_type`, `reviewed_at`, `adjudication_status`, and `notes`. Unknown identity/condition values
are written as `UNKNOWN`, not inferred. Every event must have `reviewer_type=HUMAN`, be human
`APPROVED`, timezone-stamped and
`FINAL` before freezing. Negative intervals are annotated explicitly in the source annotation
package. Two independent reviews and an adjudication record are recommended for ambiguous
boundaries.

Business-rule cases include driver/passenger/mounted/unknown phone, driver and passenger
unfastened belt, occlusion, outside-vehicle people, motorcycle prohibition, lost context reset,
hysteresis and classifier/detector conflict.

## Evaluation

Freeze truth before generating predictions. Report PHONE and NO_SEATBELT precision, recall and F1;
false events per minute/hour; misses; duplicates; time-to-detection; start latency; and event
fragmentation. Break down by declared day/night, camera, vehicle, occlusion and distance metadata.
Missing metadata is `UNKNOWN`, never inferred.

Freeze reviewed truth with `training.freeze_event_ground_truth`. It binds the CSV SHA-256 to the
immutable external-test lock and refuses overwrite. `training.evaluate_events` remains `NOT_RUN`
until frozen truth, predictions and an ACTIVE governed model lock are supplied. It verifies the
truth/external/model-lock provenance and refuses to overwrite the first frozen result.
