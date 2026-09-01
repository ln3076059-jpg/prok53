# API

## V2 runtime and evidence

- `GET /health` reports model activation state, model/config hashes, explicit fusion mode,
  fusion artifact hash/threshold, cabin-localizer availability, and streaming status.
- `GET /statistics` includes the same explicit fusion runtime mode and fail-closed state beside
  operational event counts; counts are not accuracy metrics.
- `GET /events/{event_id}` returns the event plus vehicle type, fusion/temporal scores,
  evidence links, evidence trace, and append-only review history.
- `GET /events/{event_id}/evidence/original`
- `GET /events/{event_id}/evidence/annotated`
- `GET /events/{event_id}/evidence/clip`
- `GET /events/{event_id}/evidence/trace`

All event/evidence endpoints require a bearer token. Missing evidence returns `404`; the API does
not substitute placeholder images or synthetic trace data.
`PATCH /events/{event_id}/review` returns `422` when a reviewer attempts `CONFIRMED` without a
complete temporal clip, canonical/legacy keyframes, readable trace, matching database path/hash,
and matching declared SHA-256 values. `REJECTED` and `NEEDS_REVIEW` remain available for
incomplete or altered cases.

`evidence.integrity_anchor` distinguishes new `FULL_PACKAGE_SHA256` records from readable legacy
records that anchor only the original image. Only the full-package state can become confirmation
ready.

FastAPI exposes login, video upload/detail/analysis, job status, event search/detail/review, statistics, and CSV export. OpenAPI documentation is available at `/docs`. All operational endpoints except health and login require a bearer token.
