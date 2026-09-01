# Multi-model V2 training

V2 is separate from the running `MC_BOOTSTRAP_001` experiment. Never resume the V1
checkpoint with V2 data or configuration.

## Why V2 is split by task

The old joint detector mixes phone-oriented images and seatbelt-oriented images. A phone-only
image can visibly contain an occupant or belt state without a corresponding seatbelt label;
ordinary YOLO training then treats that unlabelled evidence as background. V2 removes this
partial-label conflict:

1. `phone_detector` detects the physical phone only.
2. `seatbelt_detector` detects a comparable occupant upper-body ROI, independent of belt state.
3. `seatbelt_classifier` classifies the ROI as `seatbelt_fastened`,
   `seatbelt_unfastened`, or `uncertain_or_occluded`.
4. Vehicle context, occupant role, pose geometry, temporal persistence, and logistic fusion
   decide whether an observation can become an event.

`uncertain_or_occluded` is fail-closed. An invisible belt is never automatically converted to
`seatbelt_unfastened`.

## Data already prepared

Generate the machine-readable readiness and remediation report before any bundle build:

```powershell
py -m training.audit_v2_readiness
```

The outputs are `reports/v2_training_readiness.json` and `.md`. Readiness is deliberately split
into proposal-detector, governed-training, and production-activation levels. A PASS integrity
audit cannot promote either of the latter two levels.

Prepare the concrete human-review queues for every known gap:

```powershell
py -m training.prepare_v2_data_gaps
```

This creates group-representative phone-positive/negative queues, the exact 4,868-image
seatbelt ROI/state queue, a 500-candidate uncertainty queue compatible with the reviewer UI,
and a 2,704-group external ADT queue. The source audit is stored under
`reports/adt_seatbelt_v1/`; source proposals remain excluded until explicit decisions are
materialized.

- `datasets/derived/seatbelt_v2_balanced`: 4,868 unique source images and 4,929 upper-body
  boxes. Despite the legacy directory name, V2 adds no duplicates or synthetic oversampling.
- `reports/seatbelt_v2/audit.json`: PASS, with zero SHA, video/source group, effective group,
  or near-cluster overlap across train/val/test.
- `datasets/derived/seatbelt_classifier_v2`: 4,929 crops, preserving source splits. It is
  intentionally `ready_for_training: false` because no reviewed `uncertain_or_occluded`
  samples exist yet.
- `datasets/manifests/v2_hard_review_queue.json`: 2,273 group-diverse hard-example candidates.
  Every item remains `PENDING`; none is automatically used as a negative.
- `datasets/manifests/seatbelt_uncertain_review_v2.json`: 500 difficult upper-body boxes,
  limited to one per source video group (300 train, 100 val, 100 test), prioritized by low light,
  low contrast, blur, and border truncation.

Current upper-body crop counts are:

| Split | Fastened | Unfastened | Uncertain |
|---|---:|---:|---:|
| train | 2,441 | 1,219 | 0 |
| val | 361 | 284 | 0 |
| test | 322 | 302 | 0 |

Do not train the three-state classifier as a governed or production model until the uncertain
column contains independently captured, human-reviewed examples in every split. The separate
model-assisted pending-approval lane described below can train an explicitly exploratory classifier; it does not
clear this governed-data requirement.

Record uncertainty decisions as append-only JSONL with `candidate_id`, `decision`, `reviewer`,
and `reviewed_at`, then materialize only explicit uncertainty decisions:

```powershell
py -m training.apply_uncertain_decisions `
  datasets/manifests/seatbelt_uncertain_decisions_v2.jsonl
```

## Add genuinely independent data

Follow `datasets/v2_capture_policy.yaml`. Every sample records video, vehicle, person, camera,
conditions, occupant role, upper-body box, visible state, and SHA-256. Do not count provider
augmentations or adjacent frames as new diversity.

Validate a capture manifest before ingestion:

```powershell
py -m training.validate_v2_capture datasets/manifests/v2_capture.jsonl
```

Seatbelt ROI hard negatives use a separate proposal-only queue because an empty model output is
not a trustworthy negative label. Capture the scenarios in
`datasets/v2_seatbelt_hard_negative_policy.yaml`, then build the review queue with:

```powershell
py -m training.build_seatbelt_hard_negative_queue `
  datasets/manifests/v2_seatbelt_hard_negative_capture.jsonl `
  --require-complete-coverage
```

The builder verifies file hashes, rejects automatic approval/training labels and duplicate image
bytes, limits repeated source groups, and reports missing scenario coverage. Every emitted item
remains `PENDING`; a human must decide whether it is a true hard negative, contains an occupant
that needs re-annotation, is uncertain, or must be rejected.

The validator fails when the same video, vehicle, person, or byte-identical image crosses
splits, when group-diversity minimums are not met, when required adverse conditions are absent,
or when a derived/duplicated sample is declared.

Three licensed candidates are recorded in `datasets/sources.yaml`: c3rl seatbelt-detection-v2,
ADT Seat_belt_detection v1, and Traffic Violations Seatbelt Detection v3. They remain candidates:
their labels, capture groups, real/synthetic domain, and provenance must be reviewed before use.
The Roboflow sources require `ROBOFLOW_API_KEY`; the key must stay outside the repository.

## Build V2 datasets

The upper-body dataset has already been built and audited. To reproduce it:

```powershell
py -m training.build_seatbelt_v2 `
  --source datasets/derived/mc_bootstrap_v2_6500 `
  --output datasets/derived/seatbelt_v2_balanced

py -m training.audit_specialist_dataset `
  datasets/derived/seatbelt_v2_balanced `
  --output reports/seatbelt_v2/audit.json
```

After human review creates a group-safe uncertain JSONL, build the final classifier dataset in
a new directory. Each uncertain row needs `sample_id`, `split`, `image_path`, `yolo`,
`source_group_id`, and `effective_group_id`.

```powershell
py -m training.build_seatbelt_classifier `
  --source datasets/derived/seatbelt_v2_balanced `
  --uncertain-manifest datasets/manifests/seatbelt_uncertain_v2.jsonl `
  --output datasets/derived/seatbelt_classifier_v2_ready
```

Update the classifier `data` path in `experiments/MULTIMODEL_V2/config.yaml` to the ready
directory. The runner refuses a dataset without all three classes in train, val, and test.

## Train

### Model-assisted pending-approval detector bundle

The model-assisted pretrain pass has triaged all prepared queues without impersonating a human
reviewer. Reproduce the pending-approval lane, materialize only training-eligible core samples,
and run the fail-closed audit with:

```powershell
py -m training.prepare_v2_pretrain_review
py -m training.build_pretrain_pending_v2
py -m training.audit_pretrain_pending_v2
py -m training.build_v2_pretrain_bundle
```

The current `kaggle/MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL_PORTABLE.zip` contains the phone
detector, upper-body ROI detector, three-state seatbelt classifier, and provenance-checked
`yolo11s.pt` plus `yolo11s-cls.pt`. A model-assisted visual semantic overlay contributes 140
new genuine uncertain/occluded ROIs, bringing the classifier to the required 100/25/25
train/val/test counts. The bundle needs no Internet to obtain either base weight. This opens
exploratory training only; the overlay remains pending human approval and cannot clear the
governed-training gate.
After checksum verification and extraction on the GPU machine:

```powershell
.\START_V2_PRETRAIN_REVIEW_TRAINING.ps1 -Mode preflight
.\START_V2_PRETRAIN_REVIEW_TRAINING.ps1 -InstallDependencies `
  -BackupDirectory E:\roadwatch-v2-pretrain-backup
```

Full training runs a one-epoch smoke suite first, selects a conservative RTX 5060 Ti profile,
disables detector multi-scale, and safely resumes only a matching plan with `last.pt`. Its
outputs must remain named proposal/bootstrap or model-assisted pending-approval exploratory
weights. This lane does not authorize governed metrics, frozen external-test claims, or
production activation.

### Governed portable bundle (build later)

Build the transfer package on the preparation machine:

```powershell
py -m training.download_base_weights
py -m training.build_v2_portable_bundle
```

After governed data approval and regeneration, extract `kaggle/MULTIMODEL_V2_PORTABLE.zip` on
the GPU machine:

```powershell
# Governed training is the safe default and refuses the current PENDING bundle.
.\START_V2_TRAINING.ps1 -InstallDependencies -BackupDirectory E:\roadwatch-v2-backup

# Explicit bootstrap/proposal experiment only; never report as production accuracy.
.\START_V2_TRAINING.ps1 -AllowProposalTraining -BackupDirectory E:\roadwatch-v2-backup

# Later runs need only this command. Repeating it safely resumes matching last.pt files.
.\START_V2_TRAINING.ps1

# Optional checks.
.\START_V2_TRAINING.ps1 -Mode preflight
.\START_V2_TRAINING.ps1 -Mode smoke -Working runs/smoke
```

The package builder deletes/refuses its output when the ZIP exceeds 1 GiB. The previous
detector-only artifact is intentionally not retained because the active pending-approval bundle
already contains all three exploratory components. A governed bundle should be regenerated only
after every required class and split has human-approved evidence.
Verify the transferred ZIP against `MULTIMODEL_V2_PORTABLE.sha256` before extracting it.
The bundle includes the provenance-checked YOLO11s base weight and deterministic RTX 5060 Ti 8 GB/
16 GB profiles. `--profile auto` is the default; use `--only phone_detector` or
`--only seatbelt_detector` with separate working directories for isolated experiments.
`TRAINING_READINESS.json` is machine-readable and lists the two trainable proposal detectors
plus the blocked classifier and its exact per-split class counts. The runner auto-resumes by
default only when the immutable plan matches and an optimizer-bearing `last.pt` exists; use
`--no-resume` to require a fresh working directory.

Full training runs a one-epoch smoke experiment for every selected component first. This checks
CUDA execution and the selected VRAM profile on the target machine; use `-SkipSmoke` only after
that exact environment has already passed. `-BackupDirectory` mirrors optimizer-bearing
`last.pt`, `best.pt`, results, arguments, plan, and verified metadata after every model-save
event. If the working disk is lost, reuse the same backup directory and working path: restoration
is allowed only when both the immutable plan fingerprint and checkpoint SHA-256 match.
The launcher requires governed readiness by default. `-AllowProposalTraining` is an explicit,
auditable downgrade for bootstrap experiments and does not clear any data blocker.

The seatbelt ROI detector uses 1280 px, mild geometry/color augmentation, and no offline image
duplication. Safe base defaults use batch 1; the 16 GB portable profile raises it to batch 4.
The base config and both portable profiles disable multi-scale to prevent transient VRAM spikes.
Do not mutate a run after it starts.

```powershell
# Train one component at a time, recommended for recoverability.
py -m training.train_multimodel_v2 --working runs/v2 --only seatbelt_detector
py -m training.train_multimodel_v2 --working runs/v2_phone --only phone_detector
py -m training.train_multimodel_v2 --working runs/v2_classifier --only seatbelt_classifier
```

Training evaluates validation only. Frozen test remains unopened until every component,
threshold, camera ROI, and fusion artifact is locked.

## Calibrate and evaluate

Create validation score CSVs and calibrate per class:

```powershell
py -m training.calibrate_thresholds reports/v2_validation_scores.csv `
  --model models/candidates/v2/phone_detector.pt `
  --validation-manifest datasets/manifests/v2_validation.jsonl `
  --output reports/v2_threshold_calibration.json
```

Train the portable balanced logistic fusion model from event-level train/validation features:

```powershell
py -m training.train_fusion reports/v2_fusion_features.csv `
  --development-manifest datasets/manifests/v2_event_features_development.jsonl `
  --output models/fusion/no_seatbelt_v2.json `
  --target NO_SEATBELT --minimum-recall 0.75
```

Evaluate components per class. Test evaluation requires `reports/model_lock_v2.json`:

```powershell
py -m training.evaluate_v2 models/active/v2/seatbelt_detector.pt `
  --task detect --data datasets/derived/seatbelt_v2_balanced/data.yaml --split val

py -m training.evaluate_v2 models/active/v2/seatbelt_classifier.pt `
  --task classify --data datasets/derived/seatbelt_classifier_v2_ready --split val
```

Report phone, upper-body ROI, all three classifier states, and end-to-end event metrics
separately. Detector mAP is not violation accuracy.

## Runtime activation

The lightweight COCO vehicle and pose checkpoints are installed under `models/auxiliary/`.
After training, install the three specialist weights and activate V2:

```text
models/active/v2/phone_detector.pt
models/active/v2/seatbelt_detector.pt
models/active/v2/seatbelt_classifier.pt
```

```powershell
$env:MODEL_CONFIG_PATH = "models/model_config_v2.yaml"
uvicorn backend.main:app --reload
```

The runtime accepts cabin crops directly. Raw traffic scenes are accepted only when the local
vehicle detector is available; safety inference then runs inside each tracked vehicle ROI.
Phone events require the driver role, multi-frame persistence, and pose context; passenger or
mounted/static phones do not become violations. Classifier uncertainty or detector/classifier
disagreement becomes `NEEDS_REVIEW` or no automatic no-seatbelt event.
