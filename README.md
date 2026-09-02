# Roadwatch Driver Safety

Evidence-first driver-safety research system with an immutable single-detector V1 baseline and an
independent fail-closed V2 multi-stage architecture.

Canonical classes:

```text
0 phone
1 seatbelt_fastened
2 seatbelt_unfastened
```

The repository provides the governed workflow, runtime boundaries, review UI, evaluation tools,
and runnable application scaffold. It does **not** claim governed V2 data, final trained V2
weights, external-test results, event accuracy, human approval, or production readiness.

## Quick start

```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements-dev.txt
copy .env.example .env
python tools/create_admin.py reviewer@example.org a-strong-local-password
uvicorn backend.main:app --reload
```

In another terminal:

```bash
cd frontend
npm install
npm run dev
```

The API defaults to SQLite for local development. Set `DATABASE_URL` to the supplied MySQL connection for deployment.

## Governed workflow

1. Review the source plan and explicitly download a licensed entry with `training.download_source`.
2. Ingest immutable raw assets and build a semantics-safe human review queue.
3. Approve only vehicle-associated physical phones and explicit upper-body belt states; apply append-only decisions.
4. Split by vehicle/subject/clip group while balancing class instances.
5. Audit leakage, run CV quality analysis, and create only train-side camera-condition derivatives.
6. Freeze test, build the private Kaggle bundle, and run the one MC_001 notebook.
7. Validate, calibrate, lock, run frozen test once, then install `best.pt`.

## Model detection metrics

Status: **NOT_RUN**. No trained model exists. Validation and frozen-test results must be generated from the governed experiment and stored under `reports/`.

## System event metrics

Status: **NOT_RUN**. No independent human event ground truth exists. Detector mAP is not event accuracy.

## Project map

- `training/`: immutable ingestion, label validation, group split, audit, freeze, bundle, and evaluation
- `tools/annotation_reviewer/`: human semantic review application
- `backend/`: FastAPI, central detector, association, temporal engine, persistence, evidence, API
- `frontend/`: React/TypeScript operations dashboard using Carbon
- `notebooks/`: canonical five-stage Kaggle notebook
- `experiments/MC_001/`: sole training configuration
- `docs/`: research, semantics, architecture, evaluation, API, and deployment

Start with [dataset sources](docs/data-sources.md), [dataset governance](docs/dataset.md), [annotation semantics](docs/annotation-guide.md), [training strategy](docs/training-strategy.md), and [Kaggle training](docs/kaggle-training.md).

The independent multi-model successor is documented in [V2 training](docs/v2-training.md) and
[V2 architecture](docs/V2_ARCHITECTURE.md). V2 adds vehicle type/tracking, explicit windshield
or cabin localization, confidence-bearing occupant association, per-vehicle behavior tracking,
three-state belt classification, temporal fusion, evidence clips/traces, and append-only review.
It does not modify the running V1 experiment.

Current V2 status is **UNTRAINED / NOT APPROVED / NOT PRODUCTION READY**. See the
[senior AI audit](docs/SENIOR_AI_AUDIT.md), [limitations](docs/V2_LIMITATIONS.md),
[production readiness](docs/V2_PRODUCTION_READINESS.md), and
[event evaluation contract](docs/EVENT_EVALUATION.md).

Useful V2 audit commands remain evidence-bound:

```powershell
# Produces NOT_RUN until a real target-hardware video and weights are supplied.
py -m tools.benchmark_runtime --config models/model_config_v2.yaml

# Produces NOT_RUN until independently reviewed association annotations are supplied.
py -m training.evaluate_associations

# Measures actual manifest metadata/resolutions; missing identities remain UNKNOWN.
py -m training.audit_v2_diversity

# Rebuilds the model-assisted priority order; all 3,013 rows remain PENDING.
py -m training.prioritize_phone_negative_review

# With no real inputs, event metrics remain NOT_RUN. A real run additionally requires frozen
# reviewed truth and an ACTIVE governed model lock; the result cannot be overwritten.
py -m training.evaluate_events
```

The current diversity audit is `NOT_GOVERNED`: phone and upper-body proposal data each use one
provider and do not declare camera, video, vehicle or person identity. Subject-disjoint status is
therefore `NOT_PROVABLE`. See [the diversity audit](reports/V2_DATA_DIVERSITY_AUDIT.md),
[human review plan](docs/V2_HUMAN_REVIEW_PLAN.md),
[external-test protocol](docs/V2_EXTERNAL_TEST_PROTOCOL.md),
[calibration protocol](docs/V2_CALIBRATION_PROTOCOL.md),
[model-lock protocol](docs/V2_MODEL_LOCK_PROTOCOL.md), and
[event ground-truth protocol](docs/V2_EVENT_GROUND_TRUTH_PROTOCOL.md).

Set `APP_ENV=production` only after producing an ACTIVE, human-approved lock whose config and
component hashes match the runtime. The current untrained V2 configuration intentionally aborts
production startup.

## V2 portable GPU handoff

For the completed model-assisted pending-approval exploratory lane, transfer
`kaggle/MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL_PORTABLE.zip` plus its `.sha256`, extract it, and run:

```powershell
.\START_V2_PRETRAIN_REVIEW_TRAINING.ps1 -Mode preflight
.\START_V2_PRETRAIN_REVIEW_TRAINING.ps1 -InstallDependencies `
  -BackupDirectory E:\roadwatch-v2-pretrain-backup
```

This build contains the phone detector, upper-body ROI detector, and three-state seatbelt
classifier, plus verified offline `yolo11s.pt` and `yolo11s-cls.pt` base weights. The classifier
uncertainty gate is met at 100/25/25 train/val/test samples and the extracted bundle preflight
passes all 20,679 file hashes. Its audit state remains `MODEL_ASSISTED_PENDING_APPROVAL`:
exploratory training is allowed, while human verification, governed training, frozen external
testing, calibration, and production activation remain false until performed.

The governed V2 builder and launcher remain in the repository for later use after human
verification. Its old detector-only ZIP artifact was removed because the three-component
pending-approval bundle above supersedes it. Regenerate a governed bundle when its data gate is
actually satisfied, then run:

```powershell
.\START_V2_TRAINING.ps1 -InstallDependencies -BackupDirectory E:\roadwatch-v2-backup
```

After an interruption, repeat the same command without changing the working directory. The
runner safely resumes only a matching plan with `last.pt`.
The launcher runs a one-epoch GPU smoke test before full training and mirrors resumable
checkpoints to the optional second-disk backup directory after every model-save event.
The command above now requires governed readiness and intentionally blocks the current PENDING
bundle. Use `-AllowProposalTraining` only for an explicitly named bootstrap experiment; it does
not make the resulting weights production-ready. Run `py -m training.prepare_v2_data_gaps` to
generate the review queues and remediation plan needed to clear the gate legitimately.
