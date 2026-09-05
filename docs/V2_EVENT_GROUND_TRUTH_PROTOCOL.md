# V2 Event Ground-Truth Protocol

Status: **BLOCKED_BY_DATA / NOT_RUN**.

## Video set

Use real clips covering normal driving, driver phone use, phone near face, passenger phone,
mounted phone, fastened/unfastened/occluded belt, reflections, low light, vehicle entry/exit and
multiple vehicles. Synthetic-only clips may test logic but cannot establish production accuracy.

## Two frozen truth artifacts

`event_truth.csv` is sparse and is used only for event TP/FP/FN/F1, duplicate alerts and
time-to-detection. Each row has `video_id`, `event_id`, `event_type`, `start_seconds`,
`end_seconds`, `occupant_id`, `vehicle_id`, `cabin_id`, `occupant_role`, `label`, `visibility`,
`conditions`, `human_review_status`,
`reviewer_id`, `reviewer_type`, `reviewed_at`, `adjudication_status`, and `notes`. Unknown identity/condition values
are never accepted for `occupant_id`, `vehicle_id`, or `cabin_id` in frozen evaluation. Every
event must have `reviewer_type=HUMAN`, be human
`APPROVED`, timezone-stamped and
`FINAL` before freezing.

`context_truth.csv` is independent, human-reviewed truth for the seven safety invariants. It has
one or more half-open `[start_seconds, end_seconds)` rows per stable occupant and includes
`context_id`, `occupant_id`, `vehicle_id`, `cabin_id`, role, inside/outside state, motorcycle
state, `phone_state`, `seatbelt_state`, visibility, conditions, review provenance, and
`timeline_end_seconds`. For every external-test video, every declared occupant must be covered
continuously from 0 through the declared timeline end with no gap or overlap. A stable occupant
identity may map to only one vehicle/cabin pair within a video.

The sequence annotation schema expresses the same rule with `context_intervals` using half-open
frame ranges `[start_frame, end_frame)`. The validator rejects missing identity, missing occupant
coverage, gaps, overlaps, contradictory inside/outside context, and a final interval that does
not end at `frame_count`.

Generate both CSV artifacts from the same reviewed sequence package:

```powershell
py -m training.build_event_truth_from_sequences `
  datasets/event_sequences `
  reports/event_truth.csv `
  --context-output reports/context_truth.csv
```

Business-rule cases include driver/passenger/mounted/unknown phone, driver and passenger
unfastened belt, occlusion, outside-vehicle people, motorcycle prohibition, lost context reset,
hysteresis and classifier/detector conflict.

## Evaluation

Freeze both truth artifacts before generating predictions. Report PHONE and NO_SEATBELT precision, recall and F1;
false events per minute/hour; misses; duplicates; time-to-detection; start latency; and event
fragmentation. Break down by declared day/night, camera, vehicle, occlusion and distance metadata.
Missing identity is not treated as a wildcard. A prediction must match context by exact
`video_id`, `occupant_id`, `vehicle_id`, and `cabin_id`, and its complete interval must have one
unambiguous context signature. Otherwise the safety counters are `NOT_EVALUABLE` and frozen final
evaluation refuses to publish. A prediction over an `unknown` role, `UNKNOWN` phone state, or
`UNCERTAIN_OR_OCCLUDED` seatbelt state is likewise not evaluable rather than silently counted as
zero violations.

Freeze reviewed truth with `training.freeze_event_ground_truth`. It binds the CSV SHA-256 to the
immutable external-test lock. Freeze context separately with
`training.freeze_context_ground_truth`; its lock records the proven per-video duration and full
coverage status. `training.evaluate_events` remains `NOT_RUN` until both frozen truth artifacts,
predictions and an ACTIVE governed model lock are supplied. It verifies all hashes and refuses to
overwrite the first frozen result. `--video-minutes` must equal the duration recorded by the
frozen context lock.
