"""Acquire a specified DMD subject from untracked .dmd_urls.json."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from training.dmd.download_dmd import download_and_extract_dmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("subject_key", help="Key in .dmd_urls.json, e.g. gZ-36 or gZ-37")
    args = parser.parse_args()

    urls_file = Path(".dmd_urls.json")
    if not urls_file.exists():
        print("Error: .dmd_urls.json not found.", file=sys.stderr)
        sys.exit(1)

    with urls_file.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if args.subject_key not in data:
        print(f"Error: subject {args.subject_key} not found in .dmd_urls.json", file=sys.stderr)
        sys.exit(1)

    info = data[args.subject_key]
    out_dir = Path("datasets/external_dmd/original/extracted")
    manifest_p = Path("datasets/external_dmd/manifests/dmd_sources.jsonl")

    res = download_and_extract_dmd(
        url=info["url"],
        archive_filename=info["archive"],
        output_dir=out_dir,
        manifest_path=manifest_p,
        cleanup_archive=True,
        expected_size=info.get("expected_size"),
    )
    print(f"Subject {args.subject_key} acquired successfully.")


if __name__ == "__main__":
    main()
