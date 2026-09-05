# V2 External Test Protocol

Status: **BLOCKED_BY_DATA / EXTERNAL_TEST_NOT_RUN**.

## Isolation

The external set must use providers, cameras, videos, vehicles and capture sessions absent from
training, validation, threshold calibration and fusion training. Person identity is also disjoint
when trustworthy metadata exists; otherwise subject-disjoint status is `NOT_PROVABLE`.

Required condition coverage and minimum independent groups are defined in
`datasets/v2_external_test_policy.yaml`. Every video and event annotation needs a real file hash,
human reviewer identity, explicit `reviewer_type: HUMAN`, and timezone-aware review timestamp.

## Freeze sequence

1. Capture and annotate external video without inspecting final-model predictions.
2. Complete human semantic review and verify source/camera/video/vehicle/person identifiers.
3. Freeze the external manifest with `training.freeze_external_test`; the command refuses to
   overwrite an existing freeze artifact.
4. Freeze the independently reviewed sparse event CSV with `training.freeze_event_ground_truth`.
5. Freeze independently reviewed, gapless timeline context with
   `training.freeze_context_ground_truth`.
6. Lock the development model, thresholds and fusion artifact before generating predictions or
   reading evaluation results.
7. Run `training.evaluate_events` once; it verifies the ACTIVE model lock and all frozen hashes.
8. Preserve the result even if a gate fails.

```powershell
py -m training.freeze_external_test `
  datasets/manifests/v2_external_test.jsonl `
  datasets/manifests/v2_development_identity.jsonl

py -m training.freeze_event_ground_truth `
  reports/event_truth.csv `
  datasets/manifests/v2_external_test_frozen.json

py -m training.freeze_context_ground_truth `
  reports/context_truth.csv `
  datasets/manifests/v2_external_test_frozen.json
```

The frozen set must never select epochs, thresholds, augmentations, fusion weights or architecture.
After looking at its results, remediation requires a new experiment identity; the original frozen
result remains immutable. A corrected or expanded external set uses a new versioned output path;
neither freeze command overwrites its prior lock.
