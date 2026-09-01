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

## Level 3 — event metrics

`training.evaluate_events` accepts frozen human truth and system prediction CSV files with:

```text
video_id,event_type,start_seconds,end_seconds
```

Run only after the ground truth is independently reviewed:

```powershell
py -m training.evaluate_events `
  reports/event_truth_frozen.csv `
  reports/event_predictions.csv `
  --video-minutes 180 `
  --output reports/event_evaluation.json
```

The report contains per-event precision, recall, F1, false events/minute, missed events, duplicate
events, and mean time-to-detection. Without both files the tool reports `NOT_RUN`.

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
is unmet.
