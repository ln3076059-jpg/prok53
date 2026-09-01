# Roadwatch Driver Safety

Greenfield, reproducible three-class driver-safety system built around one YOLO11s detector and one canonical Kaggle experiment.

Canonical classes:

```text
0 phone
1 seatbelt_fastened
2 seatbelt_unfastened
```

The repository provides the full governed workflow and runnable application scaffold. It does **not** contain approved data, trained weights, human review decisions, or invented metrics.

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

The independent multi-model successor is documented in [V2 training](docs/v2-training.md). V2 separates phone detection from phone-use evidence and separates upper-body detection from three-state seatbelt classification; it does not modify the running V1 experiment.

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
