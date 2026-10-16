"""Exercise real FastAPI startup against an isolated temporary database."""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path


def run(output: Path = Path("reports/ai85_plus/backend_smoke.json")) -> dict:
    with tempfile.TemporaryDirectory(prefix="roadwatch-ai85-plus-smoke-") as folder:
        root = Path(folder)
        os.environ["DATABASE_URL"] = f"sqlite:///{(root / 'smoke.db').as_posix()}"
        os.environ["UPLOAD_DIR"] = str(root / "uploads")
        os.environ["EVIDENCE_DIR"] = str(root / "evidence")
        os.environ["APP_ENV"] = "testing"
        from fastapi.testclient import TestClient

        from backend.database import engine
        from backend.main import app

        with TestClient(app) as client:
            live = client.get("/health/live")
            ready = client.get("/health/ready")
            denied = client.get("/events")
            schema = client.get("/openapi.json")
            assert live.status_code == 200
            assert ready.status_code == 200
            assert denied.status_code in (401, 403)
            assert schema.status_code == 200
            result = {
                "status": "PASS",
                "liveness_http": live.status_code,
                "readiness_http": ready.status_code,
                "unauthenticated_events_http": denied.status_code,
                "openapi_http": schema.status_code,
                "route_count": len(schema.json()["paths"]),
                "database": "isolated temporary SQLite",
                "environment": "testing",
                "production_approval_granted": False,
            }
        engine.dispose()
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    return result


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports/ai85_plus/backend_smoke.json"))
    print(json.dumps(run(parser.parse_args().output)))
