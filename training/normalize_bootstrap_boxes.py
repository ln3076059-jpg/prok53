from __future__ import annotations

import argparse
import json
from pathlib import Path

from training.build_mc_bootstrap import SPLITS, clip_yolo_box


def normalize(root: Path) -> dict:
    changed_by_sample: dict[str, int] = {}
    changed_files = 0
    for split in SPLITS:
        for label_path in sorted((root / "labels" / split).glob("*.txt")):
            output = []
            changed_boxes = 0
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if not line.strip():
                    continue
                parts = line.split()
                if len(parts) != 5:
                    raise ValueError(f"invalid label format: {label_path}")
                class_id = int(parts[0])
                clipped, changed = clip_yolo_box([float(value) for value in parts[1:]])
                changed_boxes += changed
                output.append(
                    f"{class_id} " + " ".join(f"{value:.16g}" for value in clipped)
                )
            if changed_boxes:
                label_path.write_text("\n".join(output) + "\n", encoding="utf-8")
                changed_files += 1
                changed_by_sample[label_path.stem] = changed_boxes

    manifest_path = root / "manifest.jsonl"
    manifest = [
        json.loads(line)
        for line in manifest_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    for item in manifest:
        item["clipped_boxes"] = int(item.get("clipped_boxes", 0)) + changed_by_sample.get(
            item["sample_id"], 0
        )
    manifest_path.write_text(
        "\n".join(json.dumps(item, sort_keys=True) for item in manifest) + "\n",
        encoding="utf-8",
    )
    summary_path = root / "BOOTSTRAP_MANIFEST.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    summary["source_boxes_clipped_to_image"] = int(
        summary.get("source_boxes_clipped_to_image", 0)
    ) + sum(changed_by_sample.values())
    summary_path.write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    return {
        "changed_files": changed_files,
        "changed_boxes": sum(changed_by_sample.values()),
        "manifest_samples": len(manifest),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Inset boundary-touching bootstrap boxes safely")
    parser.add_argument("root", type=Path, nargs="?", default=Path("datasets/derived/mc_bootstrap_v1"))
    args = parser.parse_args()
    print(json.dumps(normalize(args.root), indent=2))


if __name__ == "__main__":
    main()
