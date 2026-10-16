"""Reconcile existing evidence without selecting policies on consumed subjects."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np

from training.common import sha256_file
from training.dmd.adaptive_sampling import fuse_available_views
from training.dmd.ai85_audit import read_json, write_json
from training.dmd.ai85_next import DEVELOPMENT_SUBJECTS
from training.dmd.holdout_guard import is_holdout_subject
from training.dmd.run_v5_experiments import discover_external_dmd_root

OUT = Path("reports/ai85_plus")
MIGRATION_COMMIT = "43b87ee3151060a9dd2b1b2d02ca26c0a45c138e"


def migration_differences():
    migration = read_json(OUT / "evidence_diagnostics.json")["migration"]
    rows = []
    for row in migration:
        if row.get("exact_byte_match", False) or not row.get("exists", False):
            continue
        old = subprocess.check_output(["git", "show", f"{MIGRATION_COMMIT}^:{row['old']}"])
        new = Path(row["counterpart"]).read_bytes()
        entry = {
            "old_path": row["old"],
            "counterpart": row["counterpart"],
            "old_bytes": len(old),
            "local_bytes": len(new),
            "only_newline_or_outer_whitespace": old.replace(b"\r\n", b"\n").strip()
            == new.replace(b"\r\n", b"\n").strip(),
        }
        try:
            if row["old"].endswith(".jsonl"):
                a = [json.loads(x) for x in old.decode("utf-8-sig").splitlines() if x.strip()]
                b = [json.loads(x) for x in new.decode("utf-8-sig").splitlines() if x.strip()]
            elif row["old"].endswith(".json"):
                a, b = json.loads(old.decode("utf-8-sig")), json.loads(new.decode("utf-8-sig"))
            else:
                a, b = None, None
            entry["json_semantic_equality"] = a == b if a is not None else None
        except (ValueError, UnicodeError) as exc:
            entry["parse_error_type"] = type(exc).__name__
        rows.append(entry)
    write_json(
        OUT / "dataset_migration_differences.json",
        {
            "migration_commit": MIGRATION_COMMIT,
            "byte_mismatches": len(rows),
            "newline_only": sum(r["only_newline_or_outer_whitespace"] for r in rows),
            "json_semantically_equal": sum(r.get("json_semantic_equality") is True for r in rows),
            "other_differences": sum(
                not r["only_newline_or_outer_whitespace"]
                and r.get("json_semantic_equality") is not True
                for r in rows
            ),
            "files": rows,
        },
    )


def run() -> None:
    root = discover_external_dmd_root()
    temporal = read_json(OUT / "temporal/temporal_results.json")
    baseline = temporal["variants"][0]
    multiview, cases = {}, []
    historical = read_json("reports/ai85_next/fn_recovery.json")["cases"]
    for s in DEVELOPMENT_SUBJECTS:
        source = root / "derivatives" / f"v3_features_{s}.json"
        records = read_json(source)["records"]
        available, scores = {}, []
        for view in ("body", "face", "hands"):
            values = [r[f"{view}_phone_conf"] for r in records if r[f"{view}_available"]]
            available[view] = {
                "available_records": len(values),
                "positive_proposal_records": sum(v > 0 for v in values),
                "maximum_confidence": max(values) if values else None,
            }
        for r in records:
            values = {
                v: r[f"{v}_phone_conf"] if r[f"{v}_available"] else None
                for v in ("body", "face", "hands")
            }
            scores.append(fuse_available_views(values, {"body": 1.0, "face": 1.0, "hands": 1.0}))
        multiview[s] = {
            "source": source.as_posix(),
            "sha256": sha256_file(source),
            "availability": available,
            "available_view_mean_max": max(scores),
            "note": "Confidence-only cache; missing views abstain. Fusion is uncalibrated.",
        }
        with np.load(OUT / "temporal" / f"{s}_predictions.npz") as raw:
            times, probabilities, masks = raw["timestamps"], raw["probabilities"], raw["masks"]
            phone = probabilities[:, 1:5].sum(1)
        for case in (c for c in historical if c["subject_id"] == s):
            start, end = case["truth"]["start_seconds"], case["truth"]["end_seconds"]
            inside = (times >= start) & (times <= end)
            sampling = {}
            for col, variant in enumerate(temporal["variants"]):
                local = inside & masks[:, col]
                matched = any(
                    m["ground_truth"]["start_seconds"] == start
                    and m["ground_truth"]["end_seconds"] == end
                    for m in variant["per_subject"][s]["matches"]
                )
                sampling[variant["id"]] = {
                    "centered_samples_in_interval": int(local.sum()),
                    "raw_phone_probability_max": float(phone[local].max()) if local.any() else None,
                    "matched": matched,
                }
            cases.append(
                {
                    "case_id": case["case_id"],
                    "subject_id": s,
                    "truth": case["truth"],
                    "visual_review": case["visual_review"],
                    "visual_review_scope": "inherited BODY midpoint inspection; labels unapproved",
                    "geometry": "NO_FRAME_ALIGNED_PHONE_LANDMARK_ASSOCIATION",
                    "lighting_angle_phone_pixel_GT": "NOT_ANNOTATED; see proposal sizes",
                    "sampling": sampling,
                }
            )
    # Check every removed legacy path against its local counterpart without restoring it.
    lines = subprocess.check_output(
        [
            "git",
            "diff",
            "--name-status",
            "--no-renames",
            f"{MIGRATION_COMMIT}^",
            MIGRATION_COMMIT,
            "--",
            "datasets/",
        ],
        text=True,
    ).splitlines()
    migration = []
    for line in lines:
        status, name = line.split("\t", 1)
        if status != "D":
            continue
        counterpart = Path("dataset") / Path(name).relative_to("datasets")
        if is_holdout_subject(counterpart):
            migration.append({"old": name, "status": "PROTECTED_CONTENT_NOT_READ"})
            continue
        row = {"old": name, "counterpart": counterpart.as_posix(), "exists": counterpart.is_file()}
        if counterpart.is_file():
            old_bytes = subprocess.check_output(["git", "show", f"{MIGRATION_COMMIT}^:{name}"])
            import hashlib

            row.update(
                {
                    "old_sha256": hashlib.sha256(old_bytes).hexdigest(),
                    "local_sha256": sha256_file(counterpart),
                }
            )
            row["exact_byte_match"] = row["old_sha256"] == row["local_sha256"]
        migration.append(row)
    write_json(
        OUT / "evidence_diagnostics.json",
        {
            "multiview": multiview,
            "false_negative_cases": cases,
            "migration": migration,
            "migration_summary": {
                "deleted_paths_checked": len(migration),
                "counterparts_present": sum(r.get("exists", False) for r in migration),
                "exact_matches": sum(r.get("exact_byte_match", False) for r in migration),
                "protected_not_read": sum(
                    r.get("status") == "PROTECTED_CONTENT_NOT_READ" for r in migration
                ),
            },
            "baseline": baseline["pooled"],
            "no_policy_search": True,
        },
    )
    print(json.dumps({"cases": len(cases), "removed_paths": len(migration)}))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUT)
    parser.add_argument("--migration-differences-only", action="store_true")
    args = parser.parse_args()
    OUT = args.output
    if args.migration_differences_only:
        if (OUT / "dataset_migration_differences.json").exists():
            raise FileExistsError("migration difference evidence already exists")
        migration_differences()
    else:
        if (OUT / "evidence_diagnostics.json").exists():
            raise FileExistsError(
                "use --output with a fresh directory to preserve evidence history"
            )
        run()
