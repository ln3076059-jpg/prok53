# Deployment

Copy `.env.example` to `.env`, replace every development secret, install the locked model, and
run `docker compose up --build`. `MODEL_CONFIG_PATH` is the single canonical model-config
environment variable; `MODEL_CONFIG` is not supported.

With `APP_ENV=production`, application startup validates the canonical class/threshold mapping,
all configured component weights, calibrated threshold status, required fusion artifact, and the
ACTIVE human-approved model lock declared by `activation_policy.model_lock_path`. Experiment id,
config SHA-256, and component weight hashes must match. Any blocker aborts startup with an explicit
error; production never silently falls back to uncalibrated artifacts.

Production requires HTTPS, restricted CORS origins, encrypted camera credentials, durable object
storage for evidence, database backups, and migration tooling. Camera streams stay disabled by
default to avoid SSRF. `WebcamSource` and `RTSPSource` are experimental interfaces only; enable
them after reconnect, allow-listing, secret handling, backpressure, and monitoring exist.

FastAPI BackgroundTasks is the demo `AnalysisQueue` adapter. Multi-camera production should run
`InferenceWorker` in a separate durable queue with GPU scheduling, retry, idempotency, and crash
recovery.
