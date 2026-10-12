"""Governed V4 experiment entry point using development data only.

The runner deliberately stops before evaluation when the cached development inputs are absent or
do not contain the context fields required to measure the V4 policy. It never falls back to a
consumed holdout or silently substitutes historical aggregate metrics.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from training.common import sha256_file, stable_json_hash
from training.dmd.holdout_guard import assert_not_consumed_holdout_for_development

DEVELOPMENT_SUBJECTS = ("gB-9", "gZ-36", "gC-14")
REQUIRED_RECORD_FIELDS = {
    "timestamp_seconds",
    "frame_index",
    "body_phone_conf",
    "face_phone_conf",
    "hands_phone_conf",
    "body_available",
    "face_available",
    "hands_available",
}


def _git_head() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip()


def preflight(derivatives_dir: Path, config_path: Path) -> dict[str, Any]:
    blockers: list[str] = []
    inputs: list[dict[str, Any]] = []
    for subject_id in DEVELOPMENT_SUBJECTS:
        assert_not_consumed_holdout_for_development(subject_id, caller_action="v4_preflight")
        path = derivatives_dir / f"v3_features_{subject_id}.json"
        if not path.is_file():
            blockers.append(f"missing cached development features: {path}")
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        records = payload.get("records", [])
        if not records:
            blockers.append(f"cached development features contain no records: {path}")
            continue
        missing = REQUIRED_RECORD_FIELDS - set(records[0])
        if missing:
            blockers.append(f"{path} missing required fields: {sorted(missing)}")
        inputs.append(
            {
                "subject_id": subject_id,
                "path": str(path),
                "sha256": sha256_file(path),
                "records": len(records),
            }
        )

    if not config_path.is_file():
        blockers.append(f"missing V4 config: {config_path}")
    return {
        "status": "READY" if not blockers else "BLOCKED",
        "created_at": datetime.now(UTC).isoformat(),
        "git_head": _git_head(),
        "dataset_role": "DEVELOPMENT_ONLY",
        "consumed_holdouts_forbidden": ["gZ-37", "gE-28"],
        "config_path": str(config_path),
        "config_sha256": sha256_file(config_path) if config_path.is_file() else None,
        "inputs": inputs,
        "input_set_sha256": stable_json_hash(inputs),
        "blockers": blockers,
        "generalization_claim": "NOT_ESTABLISHED",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Preflight governed Roadwatch V4 experiments")
    parser.add_argument(
        "--derivatives-dir", type=Path, default=Path("datasets/external_dmd/derivatives")
    )
    parser.add_argument(
        "--config", type=Path, default=Path("models/model_config_v4_candidate.yaml")
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/v4_upgrade/experiment_preflight.json"),
    )
    args = parser.parse_args()
    result = preflight(args.derivatives_dir, args.config)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))
    if result["status"] != "READY":
        raise SystemExit(2)


if __name__ == "__main__":
    main()
