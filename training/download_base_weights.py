from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
import urllib.request
from pathlib import Path

SOURCE_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11s.pt"
EXPECTED_SHA256 = "85a76fe86dd8afe384648546b56a7a78580c7cb7b404fc595f97969322d502d5"
EXPECTED_BYTES = 19_313_732
MAX_BYTES = 100 * 1024 * 1024


def download(output: Path) -> dict:
    if output.is_file():
        digest = hashlib.sha256(output.read_bytes()).hexdigest()
        if digest != EXPECTED_SHA256 or output.stat().st_size != EXPECTED_BYTES:
            raise ValueError(f"existing base weight does not match provenance: {output}")
        return {"path": str(output.resolve()), "bytes": output.stat().st_size, "sha256": digest}

    output.parent.mkdir(parents=True, exist_ok=True)
    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            prefix="yolo11s-", suffix=".part", dir=output.parent, delete=False
        ) as handle:
            temporary = Path(handle.name)
            digest = hashlib.sha256()
            total = 0
            with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
                content_length = response.headers.get("Content-Length")
                if content_length and int(content_length) > MAX_BYTES:
                    raise ValueError("base weight exceeds the download byte limit")
                while chunk := response.read(1024 * 1024):
                    total += len(chunk)
                    if total > MAX_BYTES:
                        raise ValueError("base weight exceeded the download byte limit")
                    digest.update(chunk)
                    handle.write(chunk)
        if total != EXPECTED_BYTES or digest.hexdigest() != EXPECTED_SHA256:
            raise ValueError("downloaded base weight failed provenance verification")
        os.replace(temporary, output)
        temporary = None
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
    return {"path": str(output.resolve()), "bytes": total, "sha256": EXPECTED_SHA256}


def main() -> None:
    parser = argparse.ArgumentParser(description="Download the governed YOLO11s base weight")
    parser.add_argument("--output", type=Path, default=Path("models/base/yolo11s.pt"))
    args = parser.parse_args()
    print(json.dumps(download(args.output), indent=2))


if __name__ == "__main__":
    main()
