# V2 External Test Protocol

Status: **BLOCKED_BY_DATA / EXTERNAL_TEST_NOT_RUN**.

## Isolation

The external set must use providers, cameras, videos, vehicles and capture sessions absent from
training, validation, threshold calibration and fusion training. Person identity is also disjoint
when trustworthy metadata exists; otherwise subject-disjoint status is `NOT_PROVABLE`.

Required condition coverage and minimum independent groups are defined in
`datasets/v2_external_test_policy.yaml`. Every video and event annotation needs a real file hash,
human reviewer identity, explicit `reviewer_type: HUMAN`, and timezone-aware review timestamp.

The official policy requires full-system identity scope and sequence intake validation.
Both holdout and development manifests must supply `capture_session_id`,
`physical_vehicle_group_id`, and `person_group_ids` in addition to SHA/source/camera/video.
Person lists are compared member by member; physical group IDs must persist across videos.
Missing or unknown physical lineage on either side blocks freezing. The eight-vehicle and
eight-person minima count physical groups, not video-scoped canonical IDs.

Holdout rows require `proposed_role=NEW_UNTOUCHED_HOLDOUT`, `prior_usage=NEVER_USED`,
explicit `model_predictions_seen=false`, and non-empty rights/independence evidence files
with matching `rights_evidence_sha256` and `independence_evidence_sha256`. The freezer
reruns `training.validate_sequence_intake` on its actual inputs and stores the evidence
path/hash records in the frozen artifact. Precheck reports cannot authorize a freeze.
Evidence integrity does not establish the truth or completeness of human declarations.
See `datasets/sequence_intake/v2_sequence_001/README.md` for the CSV/JSONL intake contract.

## Canonical truth and development completeness

`annotation_path` points directly to the canonical human-reviewed sequence JSON used by
`training.build_event_truth_from_sequences`, with `annotation_sha256` binding that file.
The official policy rejects the legacy JSON event-list format. The freezer runs the same
schema/semantic validator, requires HUMAN/FINAL provenance, and checks video/source/camera,
video hash, FPS/frame count and identity-manifest hash. All proven occupants must be covered.
For multiple cabins, supply `additional_sequence_annotations` as a list of
`{"path": "...", "sha256": "..."}` references to the other canonical sequence files.
The frozen artifact stores every source path/hash in `sequence_annotations`. Export both
event/context CSVs from these same files; annotators do not maintain a second event-list truth.

Before intake validation, a human reviews the complete development/previous-use lineage
and supplies an attestation JSON with these required fields:

| Field | Required value |
| --- | --- |
| `lineage_sha256` | SHA-256 of the exact development CSV/JSONL bytes |
| `human_review_status`, `reviewer_type`, `adjudication_status` | `APPROVED`, `HUMAN`, `FINAL` |
| `reviewer_id`, `reviewed_at` | Real reviewer identifier and timezone-aware timestamp |
| `completeness_status` | `COMPLETE` |
| `covered_usage` | All of `TRAIN`, `VALIDATION`, `THRESHOLD_CALIBRATION`, `TEMPORAL_CALIBRATION`, `FUSION_TRAINING`, `PREVIOUSLY_USED` |
| `completeness_evidence` | `path` and `sha256` of a non-empty inventory/reconciliation report |

The completeness evidence must document inventory sources, reconciliation and any usage
category with no data; include the consumed canonical test in previously-used data.
`freeze_development_lineage` binds the exact lineage, review source and evidence hashes.
Intake and official freeze verify this lock again, including its review and evidence files.
This records accountable human attestation; it cannot discover data omitted from every inventory.

## Freeze sequence

1. Capture and annotate external video without inspecting final-model predictions.
2. Complete human semantic review and verify source/camera/video/vehicle/person identifiers.
   Freeze the independent identity roster/manifest and the reviewed development lineage first.
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
py -m training.freeze_development_lineage `
  datasets/manifests/v2_sequence_001_development.jsonl `
  datasets/incoming/v2_sequence_001/development_completeness_review.json `
  --output datasets/manifests/v2_sequence_001_development_frozen.json

py -m training.freeze_external_test `
  datasets/manifests/v2_sequence_001_external.jsonl `
  datasets/manifests/v2_sequence_001_development.jsonl `
  --identity-manifest-lock datasets/manifests/v2_sequence_001_identity_frozen.json `
  --development-lineage-lock datasets/manifests/v2_sequence_001_development_frozen.json `
  --output datasets/manifests/v2_sequence_001_external_frozen.json

py -m training.build_event_truth_from_sequences `
  datasets/incoming/v2_sequence_001/reviewed_holdout_sequences `
  reports/v2_sequence_001_event_truth.csv `
  --context-output reports/v2_sequence_001_context_truth.csv

py -m training.freeze_event_ground_truth `
  reports/v2_sequence_001_event_truth.csv `
  datasets/manifests/v2_sequence_001_external_frozen.json `
  --identity-manifest-lock datasets/manifests/v2_sequence_001_identity_frozen.json `
  --output datasets/manifests/v2_sequence_001_event_truth_frozen.json

py -m training.freeze_context_ground_truth `
  reports/v2_sequence_001_context_truth.csv `
  datasets/manifests/v2_sequence_001_external_frozen.json `
  --identity-manifest-lock datasets/manifests/v2_sequence_001_identity_frozen.json `
  --output datasets/manifests/v2_sequence_001_context_truth_frozen.json
```

These are new versioned output paths, not artifacts supplied by the repository. Use the exact
development file reviewed and locked above; converting CSV to JSONL requires a new review/hash.
The reviewed sequence directory must contain exactly the canonical files bound by external freeze.

The frozen set must never select epochs, thresholds, augmentations, fusion weights or architecture.
After looking at its results, remediation requires a new experiment identity; the original frozen
result remains immutable. A corrected or expanded external set uses a new versioned output path;
neither freeze command overwrites its prior lock.
