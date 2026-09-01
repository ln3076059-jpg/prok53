from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

SPLITS = ("train", "val", "test")


def _crop(image: Image.Image, yolo: list[float], padding: float = 0.12) -> Image.Image:
    width, height = image.size
    x, y, box_width, box_height = (float(value) for value in yolo)
    half_width = box_width * (1 + 2 * padding) / 2
    half_height = box_height * (1 + 2 * padding) / 2
    left = max(0, int((x - half_width) * width))
    top = max(0, int((y - half_height) * height))
    right = min(width, int((x + half_width) * width))
    bottom = min(height, int((y + half_height) * height))
    if right <= left or bottom <= top:
        raise ValueError(f"invalid candidate crop: {yolo}")
    return image.crop((left, top, right, bottom)).convert("RGB")


def _tile(item: dict, index: int, size: int) -> Image.Image:
    path = Path(item["image_path"])
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source)
        crop = _crop(image, item["yolo"])
    crop.thumbnail((size, size), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (size, size + 58), "#161616")
    tile.paste(crop, ((size - crop.width) // 2, (size - crop.height) // 2))
    quality = item.get("quality", {})
    annotation = item.get("annotations", [{}])[0]
    lines = [
        f"{index:03d} {item['sample_id'][:12]}",
        (
            f"{item['split']} src={annotation.get('class_id')} "
            f"score={item['uncertainty_priority_score']:.3f}"
        ),
        f"brightness={quality.get('brightness', 0):.1f} contrast={quality.get('contrast', 0):.1f}",
    ]
    draw = ImageDraw.Draw(tile)
    font = ImageFont.load_default(size=13)
    for line_index, line in enumerate(lines):
        draw.text((5, size + 3 + line_index * 17), line, fill="#f4f4f4", font=font)
    return tile


def build(manifest: Path, output: Path, per_page: int, tile_size: int) -> dict:
    if output.exists():
        raise FileExistsError(output)
    output.mkdir(parents=True)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    rows = payload["samples"] if isinstance(payload, dict) else payload
    report = {"schema_version": 1, "manifest": str(manifest), "splits": {}}
    for split in SPLITS:
        selected = sorted(
            (item for item in rows if item["split"] == split),
            key=lambda item: (-float(item["uncertainty_priority_score"]), item["sample_id"]),
        )
        pages = []
        for page_start in range(0, len(selected), per_page):
            page_rows = selected[page_start : page_start + per_page]
            columns = 5
            rows_per_page = (len(page_rows) + columns - 1) // columns
            canvas = Image.new(
                "RGB",
                (columns * tile_size, rows_per_page * (tile_size + 58)),
                "#0b0b0b",
            )
            page_manifest = []
            for offset, item in enumerate(page_rows):
                global_index = page_start + offset + 1
                tile = _tile(item, global_index, tile_size)
                x = (offset % columns) * tile_size
                y = (offset // columns) * (tile_size + 58)
                canvas.paste(tile, (x, y))
                page_manifest.append(
                    {
                        "index": global_index,
                        "candidate_id": item["candidate_id"],
                        "sample_id": item["sample_id"],
                        "split": split,
                        "image_path": item["image_path"],
                        "source_group_id": item["source_group_id"],
                        "effective_group_id": item["effective_group_id"],
                        "yolo": item["yolo"],
                        "priority": item["uncertainty_priority_score"],
                    }
                )
            page_number = len(pages) + 1
            image_path = output / f"{split}_page_{page_number:02d}.jpg"
            json_path = output / f"{split}_page_{page_number:02d}.json"
            canvas.save(image_path, quality=92)
            json_path.write_text(
                json.dumps(page_manifest, indent=2, sort_keys=True), encoding="utf-8"
            )
            pages.append(
                {"image": str(image_path), "manifest": str(json_path), "items": len(page_rows)}
            )
        report["splits"][split] = {"candidates": len(selected), "pages": pages}
    (output / "CONTACT_SHEET_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build seatbelt uncertainty ROI contact sheets")
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("datasets/manifests/pretrain_pending_approval/v2_seatbelt_uncertain.json"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("reports/diagnostics/seatbelt_uncertain_contact_sheets_v2"),
    )
    parser.add_argument("--per-page", type=int, default=25)
    parser.add_argument("--tile-size", type=int, default=300)
    args = parser.parse_args()
    print(json.dumps(build(args.manifest, args.output, args.per_page, args.tile_size), indent=2))


if __name__ == "__main__":
    main()
