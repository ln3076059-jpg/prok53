# V2 Event Ground-Truth Protocol

Status: **BLOCKED_BY_DATA / NOT_RUN**.

## Video set

Use real clips covering normal driving, driver phone use, phone near face, passenger phone,
mounted phone, fastened/unfastened/occluded belt, reflections, low light, vehicle entry/exit and
multiple vehicles. Synthetic-only clips may test logic but cannot establish production accuracy.

## Annotation schema

Each event has `video_id`, `event_id`, `event_type`, `start_seconds`, `end_seconds`, vehicle/cabin
identity, occupant role, visibility/conditions, reviewer identity, review timestamp, notes and
adjudication state. Negative intervals are annotated explicitly. Two independent reviews and an
adjudication record are recommended for ambiguous boundaries.

Business-rule cases include driver/passenger/mounted/unknown phone, driver and passenger
unfastened belt, occlusion, outside-vehicle people, motorcycle prohibition, lost context reset,
hysteresis and classifier/detector conflict.

## Evaluation

Freeze truth before generating predictions. Report PHONE and NO_SEATBELT precision, recall and F1;
false events per minute/hour; misses; duplicates; time-to-detection; start latency; and event
fragmentation. Break down by declared day/night, camera, vehicle, occlusion and distance metadata.
Missing metadata is `UNKNOWN`, never inferred.

`training.evaluate_events` remains `NOT_RUN` until both frozen truth and predictions are supplied.
