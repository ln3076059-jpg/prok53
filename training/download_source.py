from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import shutil
import subprocess
import tempfile
import urllib.request
import zipfile
from datetime import UTC, datetime
from pathlib import Path

import yaml

from training.common import sha256_file, stable_json_hash


def load_source(registry: Path, source_id: str) -> dict:
    document = yaml.safe_load(registry.read_text(encoding="utf-8"))
    source = next((item for item in document["sources"] if item["source_id"] == source_id), None)
    if source is None:
        raise ValueError(f"unknown source_id: {source_id}")
    if source["status"].startswith("REJECTED") or source.get("license") in {None, "UNKNOWN"}:
        raise ValueError(f"source rejected by license gate: {source['reason']}")
    return source


def ensure_child(path: Path, parent: Path) -> None:
    resolved_path = path.resolve()
    resolved_parent = parent.resolve()
    if resolved_path == resolved_parent or resolved_parent not in resolved_path.parents:
        raise ValueError(f"unsafe output outside raw root: {resolved_path}")


def run_download(source: dict, destination: Path) -> None:
    spec = source.get("download", {})
    kind = spec.get("type")
    if kind == "direct_zip":
        archive = destination.parent / f"{source['source_id']}.zip"
        url = spec["url"]
        print(f"Downloading licensed archive: {url}", flush=True)
        curl = shutil.which("curl")
        if curl:
            segments = int(spec.get("segments", 1))
            expected_bytes = spec.get("expected_bytes")
            if segments > 1 and expected_bytes:
                part_size = (int(expected_bytes) + segments - 1) // segments
                jobs: list[tuple[int, int, Path]] = []
                for index in range(segments):
                    start = index * part_size
                    if start >= int(expected_bytes):
                        break
                    end = min(int(expected_bytes) - 1, start + part_size - 1)
                    jobs.append((start, end, archive.with_suffix(f".part{index:03d}")))

                def fetch_part(job: tuple[int, int, Path]) -> Path:
                    start, end, part = job
                    subprocess.run(
                        [
                            curl,
                            "--fail",
                            "--location",
                            "--retry",
                            "10",
                            "--retry-all-errors",
                            "--connect-timeout",
                            "30",
                            "--speed-time",
                            "30",
                            "--speed-limit",
                            "1024",
                            "--silent",
                            "--show-error",
                            "--range",
                            f"{start}-{end}",
                            "--output",
                            str(part),
                            url,
                        ],
                        check=True,
                    )
                    if part.stat().st_size != end - start + 1:
                        raise ValueError(f"range download size mismatch for {start}-{end}")
                    return part

                with concurrent.futures.ThreadPoolExecutor(max_workers=len(jobs)) as pool:
                    parts = list(pool.map(fetch_part, jobs))
                with archive.open("xb") as output:
                    for part in parts:
                        with part.open("rb") as source_handle:
                            shutil.copyfileobj(source_handle, output, length=1024 * 1024)
                        part.unlink()
            else:
                subprocess.run(
                    [
                        curl,
                        "--fail",
                        "--location",
                        "--retry",
                        "3",
                        "--output",
                        str(archive),
                        url,
                    ],
                    check=True,
                )
        else:
            request = urllib.request.Request(
                url, headers={"User-Agent": "driver-safety-dataset-downloader/1.0"}
            )
            with urllib.request.urlopen(request, timeout=60) as response, archive.open("xb") as handle:
                shutil.copyfileobj(response, handle, length=1024 * 1024)
        expected_bytes = spec.get("expected_bytes")
        if expected_bytes is not None and archive.stat().st_size != int(expected_bytes):
            raise ValueError(
                f"archive size mismatch: expected {expected_bytes}, received {archive.stat().st_size}"
            )
        expected_sha256 = spec.get("sha256")
        if expected_sha256 and sha256_file(archive) != expected_sha256:
            raise ValueError("archive SHA256 mismatch")
        with zipfile.ZipFile(archive) as bundle:
            destination_root = destination.resolve()
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if target != destination_root and destination_root not in target.parents:
                    raise ValueError(f"unsafe archive member: {member.filename}")
            bundle.extractall(destination)
        archive.unlink()
        return
    if kind == "kaggle":
        archive = destination.parent / f"{source['source_id']}.zip"
        url = f"https://www.kaggle.com/api/v1/datasets/download/{spec['slug']}"
        print(f"Downloading public Kaggle dataset: {url}", flush=True)
        with urllib.request.urlopen(url) as response, archive.open("xb") as handle:
            shutil.copyfileobj(response, handle, length=1024 * 1024)
        with zipfile.ZipFile(archive) as bundle:
            destination_root = destination.resolve()
            for member in bundle.infolist():
                target = (destination / member.filename).resolve()
                if target != destination_root and destination_root not in target.parents:
                    raise ValueError(f"unsafe archive member: {member.filename}")
            bundle.extractall(destination)
        archive.unlink()
        return
    if kind == "huggingface":
        try:
            from huggingface_hub import snapshot_download
        except ImportError as exc:
            raise RuntimeError("install requirements-data.txt before Hugging Face download") from exc
        snapshot_download(
            repo_id=spec["repo_id"],
            repo_type="dataset",
            revision=spec.get("revision", "main"),
            local_dir=destination,
            max_workers=int(spec.get("max_workers", 4)),
        )
        return
    if kind == "roboflow":
        api_key = os.environ.get("ROBOFLOW_API_KEY")
        if not api_key:
            raise RuntimeError("ROBOFLOW_API_KEY is required; never put it in the repository")
        try:
            from roboflow import Roboflow
        except ImportError as exc:
            raise RuntimeError("install requirements-data.txt before Roboflow download") from exc
        project = Roboflow(api_key=api_key).workspace(spec["workspace"]).project(spec["project"])
        project.version(int(spec["version"])).download(
            spec.get("format", "yolov11"),
            location=str(destination),
            # `destination` is a newly-created private staging directory. Roboflow's
            # SDK otherwise treats the existing empty directory as an already
            # downloaded dataset and silently returns without writing any files.
            overwrite=True,
        )
        return
    raise ValueError(f"source {source['source_id']} requires a manual download from {source['source_url']}")


def build_manifest(source: dict, root: Path) -> dict:
    files = {
        str(path.relative_to(root)).replace("\\", "/"): sha256_file(path)
        for path in sorted(root.rglob("*"))
        if path.is_file() and path.name != "DOWNLOAD_MANIFEST.json"
    }
    if not files:
        raise ValueError("download produced no files")
    return {
        "schema_version": 1,
        "source_id": source["source_id"],
        "source_url": source["source_url"],
        "registry_version": str(source["version"]),
        "license": source["license"],
        "downloaded_at": datetime.now(UTC).isoformat(),
        "file_count": len(files),
        "files": files,
        "tree_sha256": stable_json_hash(files),
    }


def download(source: dict, raw_root: Path) -> Path:
    final = raw_root / source["source_id"] / str(source["version"])
    ensure_child(final, raw_root)
    if final.exists():
        raise FileExistsError(f"raw source is immutable and already exists: {final}")
    raw_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=f".{source['source_id']}-", dir=raw_root) as temp:
        staged = Path(temp) / "payload"
        staged.mkdir()
        run_download(source, staged)
        manifest = build_manifest(source, staged)
        (staged / "DOWNLOAD_MANIFEST.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
        )
        final.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(staged), str(final))
    return final


def main() -> None:
    parser = argparse.ArgumentParser(description="License-gated immutable dataset downloader")
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--accept-license", required=True, help="Must exactly match the registry license")
    parser.add_argument("--registry", type=Path, default=Path("datasets/sources.yaml"))
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/raw"))
    args = parser.parse_args()
    source = load_source(args.registry, args.source_id)
    if args.accept_license != source["license"]:
        raise ValueError(
            f"license acceptance mismatch: expected {source['license']!r}, got {args.accept_license!r}"
        )
    target = download(source, args.raw_root)
    manifest = json.loads((target / "DOWNLOAD_MANIFEST.json").read_text(encoding="utf-8"))
    print(json.dumps({"raw_path": str(target), **manifest}, indent=2))


if __name__ == "__main__":
    main()
