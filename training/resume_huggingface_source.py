from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from training.download_source import build_manifest, ensure_child, load_source


def resume(source: dict, staging_root: Path, raw_root: Path, max_workers: int) -> Path:
    spec = source.get("download", {})
    if spec.get("type") != "huggingface":
        raise ValueError("resume_huggingface_source only accepts Hugging Face sources")
    staging_root = staging_root.resolve()
    raw_root = raw_root.resolve()
    ensure_child(staging_root, raw_root)
    if not staging_root.name.startswith(f".{source['source_id']}-"):
        raise ValueError(f"staging directory does not match source: {staging_root}")
    payload = staging_root / "payload"
    if not payload.is_dir():
        raise FileNotFoundError(f"missing staged payload: {payload}")
    final = raw_root / source["source_id"] / str(source["version"])
    ensure_child(final, raw_root)
    if final.exists():
        raise FileExistsError(f"final raw source already exists: {final}")

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise RuntimeError("install requirements-data.txt before Hugging Face resume") from exc
    snapshot_download(
        repo_id=spec["repo_id"],
        repo_type="dataset",
        revision=spec.get("revision", "main"),
        local_dir=payload,
        max_workers=max_workers,
    )
    manifest = build_manifest(source, payload)
    (payload / "DOWNLOAD_MANIFEST.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    final.parent.mkdir(parents=True, exist_ok=True)
    shutil.move(str(payload), str(final))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="Resume and finalize an interrupted Hugging Face source")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--staging-root", type=Path, required=True)
    parser.add_argument("--accept-license", required=True)
    parser.add_argument("--registry", type=Path, default=Path("datasets/sources.yaml"))
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/raw"))
    parser.add_argument("--max-workers", type=int, default=2)
    args = parser.parse_args()
    source = load_source(args.registry, args.source_id)
    if args.accept_license != source["license"]:
        raise ValueError(f"license acceptance mismatch: expected {source['license']!r}")
    target = resume(source, args.staging_root, args.raw_root, args.max_workers)
    manifest = json.loads((target / "DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8"))
    print(json.dumps({"raw_path": str(target), **manifest}, indent=2))


if __name__ == "__main__":
    main()
