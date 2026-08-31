from __future__ import annotations

import argparse
import json
from pathlib import Path


def load_ingest(path: Path) -> dict[str, dict]:
    records: dict[str, dict] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            item = json.loads(line)
            if item.get("sample_id"):
                records[item["sample_id"]] = item
    return records


def enrich(bootstrap_manifest: Path, ingest_manifest: Path) -> dict:
    ingest = load_ingest(ingest_manifest)
    enriched = []
    missing = []
    for line in bootstrap_manifest.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        item = json.loads(line)
        source = ingest.get(item["sample_id"])
        if source is None:
            missing.append(item["sample_id"])
        else:
            item["sha256"] = source.get("sha256")
            item["phash"] = source.get("phash")
        enriched.append(item)
    if missing:
        raise ValueError(f"missing {len(missing)} bootstrap samples in ingest manifest")
    temporary = bootstrap_manifest.with_suffix(bootstrap_manifest.suffix + ".tmp")
    temporary.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in enriched) + "\n",
        encoding="utf-8",
    )
    temporary.replace(bootstrap_manifest)
    return {"records": len(enriched), "with_sha256": len(enriched), "with_phash": len(enriched)}


def main() -> None:
    parser = argparse.ArgumentParser(description="Add source hashes to an existing bootstrap manifest")
    parser.add_argument("bootstrap_manifest", type=Path)
    parser.add_argument("ingest_manifest", type=Path)
    args = parser.parse_args()
    print(json.dumps(enrich(args.bootstrap_manifest, args.ingest_manifest), indent=2))


if __name__ == "__main__":
    main()
