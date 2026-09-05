from __future__ import annotations

import argparse
from pathlib import Path

from training.extract_identity_manifest import freeze_identity_manifest


def main() -> None:
    parser = argparse.ArgumentParser(description="Freeze and lock runtime identity manifest")
    parser.add_argument("manifest", type=Path, help="Path to identity manifest JSON")
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("datasets/manifests/v2_identity_manifest_frozen.json"),
        help="Path to output frozen identity manifest lock JSON",
    )
    args = parser.parse_args()
    lock = freeze_identity_manifest(args.manifest, args.output)
    print(f"Frozen identity manifest locked at {args.output} (SHA256: {lock['manifest_sha256']})")


if __name__ == "__main__":
    main()
