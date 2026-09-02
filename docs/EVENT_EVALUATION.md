# Event Evaluation

Detector AP is not violation accuracy. V2 evaluation has three separate levels.

## Level 1 — component metrics

- Phone detector: precision, recall, F1, AP50, AP50-95.
- Upper-body detector: precision, recall, F1, AP50, AP50-95.
- Seatbelt classifier: per-class precision/recall/F1, macro F1, confusion matrix.
- Pose: keypoint availability/visibility rate on the target camera domain.

Use `training.evaluate_v2` for detector/classifier validation. Threshold selection uses validation
only. Frozen test refuses to run until the model lock exists.

## Level 2 — association metrics

Create an independent annotation file for:

- vehicle-context correctness;
- driver/passenger/rear/unknown role correctness;
- phone-to-occupant and phone-to-driver correctness;
- track identity switches and fragmentation.

Report accuracy by camera, handedness, vehicle type, day/night, glare, blur, and occlusion. An
`unknown` prediction is not silently counted as driver.

Run the association/pose availability evaluator only on independently reviewed CSV rows:

```powershell
py -m training.evaluate_associations `
  reports/v2_association_annotations.csv `
  --output reports/v2_association_evaluation.json
```

Required columns are explicit truth/prediction pairs for vehicle context, cabin localization,
occupant role, and phone-to-driver association, plus pose/wrist/face-keypoint availability. With
no input, the checked-in report remains `NOT_RUN`.

## Level 3 — event metrics

`training.evaluate_events` accepts frozen human truth and system prediction CSV files. The metric
reader needs the four columns below; the preceding ground-truth freeze additionally requires the
review, adjudication, identity and condition columns in `docs/V2_EVENT_GROUND_TRUTH_PROTOCOL.md`.

```text
video_id,event_type,start_seconds,end_seconds
```

Freeze independently reviewed truth first:

```powershell
py -m training.freeze_event_ground_truth `
  reports/event_truth.csv `
  datasets/manifests/v2_external_test_frozen.json
```

Run only after an ACTIVE governed model lock exists:

```powershell
py -m training.evaluate_events `
  reports/event_truth.csv `
  reports/event_predictions.csv `
  --video-minutes 180 `
  --ground-truth-lock datasets/manifests/v2_event_truth_frozen.json `
  --model-lock reports/model_lock_v2.json `
  --output reports/event_evaluation.json
```

The report contains per-event precision, recall, F1, false events/minute/hour, missed events,
duplicate/fragmented events, and mean time-to-detection. It records hashes for truth, predictions,
ground-truth lock, external-test lock and model lock. It refuses changed truth, an inactive or
incomplete lock, and overwriting a prior result. Without both CSV files the tool reports `NOT_RUN`.

## External test protocol

- Source, camera, vehicles, and people should be disjoint from training where metadata permits.
- Include day, night, low light, rain, glare, reflection, blur, partial occlusion, dark windshield,
  and different camera angles.
- Freeze content and hashes before final model selection.
- Never tune thresholds, fusion, cabin geometry, or role association on the frozen test.
- Report failures and abstentions (`UNKNOWN`/`NEEDS_REVIEW`) explicitly.

Freeze only after every asset and event annotation has been human-approved:

```powershell
py -m training.freeze_external_test `
  datasets/manifests/v2_external_test.jsonl `
  datasets/manifests/v2_development.jsonl `
  --output datasets/manifests/v2_external_test_frozen.json
```

The command verifies file hashes, reviewer provenance, condition coverage, policy minimums, and
source/camera/video/hash disjointness. It fails without writing a frozen artifact when any gate
is unmet and refuses to overwrite an existing freeze.
