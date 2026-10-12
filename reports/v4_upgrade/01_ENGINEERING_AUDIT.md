# Roadwatch V4 Engineering Audit

Date: 2026-10-12. Source commit: `925e81a5a76fcf6bcadb32ba9e9633011d12e572`.
Working branch: `roadwatch-v4-candidate`.

## Findings and repairs

| Severity | Finding | Evidence | Status |
|---|---|---|---|
| High | Frontend CSV link did not attach the Bearer token | `App.tsx` used a plain `href` | Fixed with authenticated fetch, Blob download and error handling |
| High | Review/export endpoints had authentication but no role allow-list | Both depended only on `current_user` | Fixed: review allows reviewer/admin; export allows reviewer/auditor/admin |
| High | Concurrent reviewers could both write from a stale state | Read-then-write transition had no compare-and-set | Fixed with conditional SQL update and 409 conflict; regression tested |
| High | Production queue was process-local | FastAPI `BackgroundTasks` was the only adapter | Database queue and external worker added; production config requires it |
| Medium | Video upload trusted extension and MIME | Arbitrary bytes could be saved as video | MP4/MOV, AVI and EBML container signatures plus empty-file rejection added |
| Medium | CI failed Ruff F821 | `Any` missing in `run_benchmark_v2.py` | Fixed |
| Medium | README referenced nonexistent `training.dmd.run_v3_evaluation` | Module absent | Replaced with existing development-only commands and holdout warning |
| Medium | Runtime environment values were incomplete | Token duration and queue mode absent | Examples synchronized; testing and production profiles added |
| Medium | No liveness/readiness routes | No health endpoints | `/health/live` and model-aware `/health/ready` added |
| Blocker | Full suite requires tracked files currently deleted under `datasets/` | 3,558 pre-existing deletions in working tree | Not restored or overwritten |

The pre-existing dataset deletions and untracked `prompt.md` were preserved. No historical benchmark,
model lock, human review record, or ground truth was modified.

## Artifact integrity

All locked V2 component weights exist and match `models/locked/v2_baseline_001/model_lock.json`:

- phone detector: `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`
- seatbelt detector: `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500`
- seatbelt classifier: `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4`

The lock itself states `production_ready: false` and `human_verified: false`.

## Environment

- Python 3.11.9
- Node 22.17.1; npm 11.6.2
- PyTorch 2.13.0 CPU build
- CUDA unavailable; `nvidia-smi` unavailable
- Free space observed at audit start: about 8.9 GB on drive D

## Status

Engineering changes implemented and locally verified. Full CI remains blocked only in dataset-bound
test groups because the required tracked schema/policy files are absent from the current working tree.
