from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont, ImageOps

from training.run_review1_review import _resolve_image


def _tile(item: dict, position: int, tile_width: int, tile_height: int) -> Image.Image:
    path = Path(item["resolved_image_path"])
    with Image.open(path) as source:
        image = ImageOps.exif_transpose(source).convert("RGB")
    image.thumbnail((tile_width, tile_height - 58), Image.Resampling.LANCZOS)
    tile = Image.new("RGB", (tile_width, tile_height), "#111816")
    offset_x = (tile_width - image.width) // 2
    offset_y = (tile_height - 58 - image.height) // 2
    tile.paste(image, (offset_x, offset_y))
    draw = ImageDraw.Draw(tile)
    for annotation in item.get("annotations", []):
        x, y, width, height = (float(value) for value in annotation["yolo"])
        scale_x = image.width
        scale_y = image.height
        left = offset_x + (x - width / 2) * scale_x
        top = offset_y + (y - height / 2) * scale_y
        right = offset_x + (x + width / 2) * scale_x
        bottom = offset_y + (y + height / 2) * scale_y
        draw.rectangle((left, top, right, bottom), outline="#ff3b30", width=3)
    font = ImageFont.load_default(size=14)
    confidence = float(item.get("proposal_review", {}).get("confidence", 0))
    draw.text(
        (7, tile_height - 52),
        f"{position:03d}  {item['sample_id'][:22]}",
        fill="#ffffff",
        font=font,
    )
    draw.text(
        (7, tile_height - 30),
        f"boxes={len(item.get('annotations', []))}  proposal={confidence:.2f}",
        fill="#c6d8d1",
        font=font,
    )
    return tile


def build(
    manifest: Path,
    output: Path,
    *,
    reason: str | None,
    offset: int,
    limit: int,
    per_page: int,
    columns: int,
    tile_width: int,
    tile_height: int,
    datasets_root: Path,
) -> dict:
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(output)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    queue = (
        payload
        if isinstance(payload, list)
        else payload.get("samples", payload.get("selected", []))
    )
    candidates = [
        item
        for item in queue
        if reason is None or reason in item.get("proposal_review", {}).get("reason", [])
    ]
    selected = candidates[offset : offset + limit]
    output.mkdir(parents=True, exist_ok=True)
    pages = []
    for start in range(0, len(selected), per_page):
        page_items = []
        current = selected[start : start + per_page]
        rows = (len(current) + columns - 1) // columns
        sheet = Image.new("RGB", (columns * tile_width, rows * tile_height), "#07100d")
        for page_offset, raw_item in enumerate(current):
            item = dict(raw_item)
            resolved = _resolve_image(item, datasets_root)
            item["resolved_image_path"] = str(resolved)
            position = start + page_offset + 1
            tile = _tile(item, position, tile_width, tile_height)
            sheet.paste(
                tile,
                (
                    (page_offset % columns) * tile_width,
                    (page_offset // columns) * tile_height,
                ),
            )
            page_items.append(
                {
                    "index": position,
                    "sample_id": item["sample_id"],
                    "image_path": str(resolved),
                    "source_group_id": item.get("source_group_id"),
                    "annotations": item.get("annotations", []),
                }
            )
        page_number = len(pages) + 1
        image_path = output / f"page_{page_number:03d}.jpg"
        manifest_path = output / f"page_{page_number:03d}.json"
        sheet.save(image_path, quality=94)
        manifest_path.write_text(
            json.dumps(page_items, indent=2, sort_keys=True), encoding="utf-8"
        )
        pages.append(
            {"image": str(image_path), "manifest": str(manifest_path), "items": len(current)}
        )
    report = {
        "schema_version": 1,
        "source_manifest": str(manifest),
        "selection_reason": reason,
        "selection_offset": offset,
        "selected": len(selected),
        "pages": pages,
    }
    (output / "CONTACT_SHEET_REPORT.json").write_text(
        json.dumps(report, indent=2, sort_keys=True), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Build deterministic Review 1 contact sheets")
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reason")
    parser.add_argument("--offset", type=int, default=0)
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--per-page", type=int, default=20)
    parser.add_argument("--columns", type=int, default=4)
    parser.add_argument("--tile-width", type=int, default=360)
    parser.add_argument("--tile-height", type=int, default=300)
    parser.add_argument("--datasets-root", type=Path, default=Path("datasets"))
    args = parser.parse_args()
    report = build(
        args.manifest,
        args.output,
        reason=args.reason,
        offset=args.offset,
        limit=args.limit,
        per_page=args.per_page,
        columns=args.columns,
        tile_width=args.tile_width,
        tile_height=args.tile_height,
        datasets_root=args.datasets_root,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
