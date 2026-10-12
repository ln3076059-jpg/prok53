# Roadwatch V4 Final Upgrade Report

## Engineering status

The branch `roadwatch-v4-candidate` contains verified code changes for CI repair, authenticated CSV
download, RBAC, atomic review transitions, upload container validation, environment separation,
database-backed job claiming, health endpoints and documentation correction. Locked artifacts and
pre-existing dataset deletions were preserved.

## AI improvements

V4 introduces explicit phone-use context states, strict abstention, stable-occupant association when
detector track IDs change, and inspectable temporal states. The candidate shortens the occlusion
bridge to the historically tested 2 s setting and remains production-blocked. Unit/regression tests
verify policy behavior, but development accuracy could not be measured because required caches are
missing.

## Metrics before and after

Historical V3: precision 30.8%, recall 40.0%, F1 34.8%, 1.116 FA/min (TP/FP/FN 4/9/6).
V4: not run; no after metric or improvement claim is made. Seatbelt historical macro-F1 is 0.5727;
uncertain recall is 0.04.

## Performance

Historical CPU component pipeline: 0.649 FPS, p50 1254.30 ms, p95 1759.80 ms. V4 E2E, GPU, VRAM
and accuracy/speed trade-off measurements were not run due missing media and CUDA.

## Scientific validity

Holdouts `gZ-37` and `gE-28` were not used for V4 development. The experiment runner enforces the
development subject list and emitted a blocker with input/config provenance. No labels, metric
definitions, matching rules, model locks or historical reports were altered.

## Commands

```powershell
py -3.11 -m pip install -r requirements-dev.txt
py -3.11 -m ruff check --select E9,F63,F7,F82 backend training tests tools
py -3.11 -m compileall -q backend training tests tools
py -3.11 -m pytest -q
cd frontend; npm ci; npm run build; cd ..
py -3.11 training/dmd/run_v4_experiments.py
py -3.11 tools/run_analysis_worker.py
uvicorn backend.main:app --host 0.0.0.0 --port 8000
```

## Verdict

`ENGINEERING_PARTIAL`

The implemented and dataset-independent portions are verified. `AI_VALIDATED_ON_DEVELOPMENT`,
`AI_VALIDATED_ON_NEW_HOLDOUT` and `PRODUCTION_READY` are not claimed.
