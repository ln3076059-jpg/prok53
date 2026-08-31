# Roadwatch Driver Safety AI

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
