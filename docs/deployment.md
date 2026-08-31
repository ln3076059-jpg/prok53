# Deployment

Copy `.env.example` to `.env`, replace every development secret, install the locked model, and run `docker compose up --build`. Production requires HTTPS, restricted CORS origins, encrypted camera credentials, durable object storage for evidence, database backups, and migration tooling. Camera streams are disabled by default to avoid SSRF.

