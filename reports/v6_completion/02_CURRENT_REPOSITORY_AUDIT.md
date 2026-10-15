# Current Repository Audit

## State

- Branch: `roadwatch-v4-candidate`
- Baseline commit at execution: `9c0e872277a02ae99f30dbecad2195333d23423a`
- The worktree already contained extensive user-owned changes and deleted `datasets/` files. They were not restored, removed, staged, or overwritten.
- Usable governed DMD data was found under singular `dataset/external_dmd/`; code discovery supports both singular and plural roots.

## Runtime and research separation

- `backend/ai/` implements detection normalization, vehicle/cabin localization, occupant association, tracking, temporal events, evidence and fusion.
- `backend/services/` implements upload, video analysis and both background and durable database queue paths.
- `models/model_config_v2.yaml` is the current development runtime configuration. It is not marked production-approved.
- V5/V6 action models remain offline research candidates and are not activated by the API.
- `SafetyDetector.validate_runtime()` rejects production startup unless artifacts, calibration state and approval gates pass.

## Software surface

- Backend: FastAPI, JWT/RBAC, database persistence, job retry/claim logic, evidence, review and CSV export are present.
- Frontend: login, image/video upload, progress/error states, event filtering, evidence preview/video, review actions, statistics and export are present.
- Camera streams remain disabled by `.env.example`; roadside behavior is not claimed.
- Impeccable static detector returned no findings for `frontend/src`; the frontend is monolithic but functionally complete for this scope.
- CI runs on pull requests and pushes to `main`; a development-branch PR is sufficient and no push to `main` is needed.
- Full Ruff audit: 1165 historical findings; 0 in the CI critical-code set. Largest groups are E501=812, F401=74, I001=38 and UP=137.

## Verification snapshot

| Check | Result | Duration |
|---|---:|---:|
| python_compile | PASS | 0.47s |
| ruff_ci_gate | PASS | 0.39s |
| ruff_full_audit | FAIL | 0.35s |
| pytest_targeted_v5_v6 | PASS | 16.61s |
| pytest_full | FAIL | 63.32s |
| backend_import_smoke | PASS | 10.58s |
| frontend_typescript_build | PASS | 35.54s |
| docker_compose_config | NOT_AVAILABLE | 0.00s |
| pytest_backend_pipeline_expanded | PASS (112 passed, 1 skipped) | 19.68s |
