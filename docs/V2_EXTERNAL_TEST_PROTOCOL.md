# V2 External Test Protocol

Status: **BLOCKED_BY_DATA / EXTERNAL_TEST_NOT_RUN**.

## Isolation

The external set must use providers, cameras, videos, vehicles and capture sessions absent from
training, validation, threshold calibration and fusion training. Person identity is also disjoint
when trustworthy metadata exists; otherwise subject-disjoint status is `NOT_PROVABLE`.

Required condition coverage and minimum independent groups are defined in
`datasets/v2_external_test_policy.yaml`. Every video and event annotation needs a real file hash,
human reviewer identity and timezone-aware review timestamp.

## Freeze sequence

1. Capture and annotate external video without inspecting final-model predictions.
2. Complete human semantic review and verify source/camera/video/vehicle/person identifiers.
3. Lock the development model, thresholds and fusion artifact.
4. Freeze the external manifest with `training.freeze_external_test`.
5. Record the frozen manifest and policy hashes in the evaluation run.
6. Run evaluation once and preserve the result even if a gate fails.

```powershell
py -m training.freeze_external_test `
  datasets/manifests/v2_external_test.jsonl `
  datasets/manifests/v2_development_identity.jsonl
```

The frozen set must never select epochs, thresholds, augmentations, fusion weights or architecture.
After looking at its results, remediation requires a new experiment identity; the original frozen
result remains immutable.
