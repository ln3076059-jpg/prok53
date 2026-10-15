"""Run the V6 software verification matrix and save auditable command results."""

from __future__ import annotations

import json
import platform
import shutil
import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]


def _run(
    name: str,
    command: list[str],
    *,
    cwd: Path = ROOT,
    timeout: int = 900,
) -> dict[str, Any]:
    started = time.perf_counter()
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=timeout,
            check=False,
        )
        output = ((completed.stdout or "") + "\n" + (completed.stderr or "")).strip()
        return {
            "name": name,
            "command": command,
            "status": "PASS" if completed.returncode == 0 else "FAIL",
            "return_code": completed.returncode,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "output_tail": output[-12000:],
        }
    except subprocess.TimeoutExpired as exc:
        output = ((exc.stdout or "") + "\n" + (exc.stderr or "")).strip()
        return {
            "name": name,
            "command": command,
            "status": "TIMEOUT",
            "return_code": None,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "output_tail": output[-12000:],
        }
    except OSError as exc:
        return {
            "name": name,
            "command": command,
            "status": "NOT_AVAILABLE",
            "return_code": None,
            "duration_seconds": round(time.perf_counter() - started, 3),
            "output_tail": str(exc),
        }


def main() -> None:
    python = shutil.which("py") or shutil.which("python")
    if python is None:
        raise RuntimeError("Python launcher not found")
    prefix = [python, "-3.11"] if Path(python).name.lower() == "py.exe" else [python]
    npm = shutil.which("npm.cmd") or shutil.which("npm") or "npm"
    commands = [
        (
            "python_compile",
            [*prefix, "-m", "compileall", "-q", "backend", "training", "tests"],
            ROOT,
        ),
        (
            "ruff_ci_gate",
            [
                *prefix,
                "-m",
                "ruff",
                "check",
                "backend",
                "training",
                "tests",
                "--select",
                "E9,F63,F7,F82",
            ],
            ROOT,
        ),
        (
            "ruff_full_audit",
            [*prefix, "-m", "ruff", "check", "backend", "training", "tests"],
            ROOT,
        ),
        (
            "pytest_targeted_v5_v6",
            [
                *prefix,
                "-m",
                "pytest",
                "-q",
                "tests/test_v5_video_action.py",
                "tests/test_v5_result_integrity.py",
                "tests/test_v6_action_hierarchy.py",
                "tests/test_v6_seatbelt_evaluation.py",
            ],
            ROOT,
        ),
        ("pytest_full", [*prefix, "-m", "pytest", "-q"], ROOT),
        (
            "backend_import_smoke",
            [*prefix, "-c", "from backend.main import app; print(app.title)"],
            ROOT,
        ),
        ("frontend_typescript_build", [npm, "run", "build"], ROOT / "frontend"),
    ]
    results = [_run(name, command, cwd=cwd) for name, command, cwd in commands]
    if shutil.which("docker"):
        results.append(_run("docker_compose_config", ["docker", "compose", "config"]))
    else:
        results.append(
            {
                "name": "docker_compose_config",
                "command": ["docker", "compose", "config"],
                "status": "NOT_AVAILABLE",
                "return_code": None,
                "duration_seconds": 0.0,
                "output_tail": "Docker CLI is not installed or not on PATH.",
            }
        )
    payload = {
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "results": results,
        "summary": {
            status: sum(row["status"] == status for row in results)
            for status in ("PASS", "FAIL", "TIMEOUT", "NOT_AVAILABLE")
        },
    }
    output = ROOT / "reports" / "v6_completion" / "v6_test_results.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))


if __name__ == "__main__":
    main()
