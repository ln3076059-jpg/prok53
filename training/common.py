from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path

import imagehash
from PIL import Image

CLASS_NAMES = {0: "phone", 1: "seatbelt_fastened", 2: "seatbelt_unfastened"}
ALLOWED_REVIEW_STATUSES = {"APPROVED", "REJECTED", "UNCERTAIN", "PENDING"}


@dataclass(frozen=True)
class YoloLabel:
    class_id: int
    x_center: float
    y_center: float
    width: float
    height: float

    def validate(self) -> None:
        if self.class_id not in CLASS_NAMES:
            raise ValueError(f"invalid class id: {self.class_id}")
        values = (self.x_center, self.y_center, self.width, self.height)
        if any(not 0 <= value <= 1 for value in values):
            raise ValueError(f"coordinates out of range: {values}")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("box has zero area")
        if self.x_center - self.width / 2 < 0 or self.x_center + self.width / 2 > 1:
            raise ValueError("box exceeds horizontal image bounds")
        if self.y_center - self.height / 2 < 0 or self.y_center + self.height / 2 > 1:
            raise ValueError("box exceeds vertical image bounds")


def parse_yolo_line(line: str, allowed_classes: set[int] | None = None) -> YoloLabel:
    parts = line.split()
    if len(parts) != 5:
        raise ValueError(f"expected 5 fields, received {len(parts)}")
    label = YoloLabel(int(parts[0]), *(float(value) for value in parts[1:]))
    label.validate()
    if allowed_classes is not None and label.class_id not in allowed_classes:
        raise ValueError(f"class id {label.class_id} not allowed in this dataset")
    return label


def read_labels(path: Path) -> list[YoloLabel]:
    if not path.exists():
        return []
    return [parse_yolo_line(line) for line in path.read_text().splitlines() if line.strip()]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def perceptual_hash(path: Path) -> str:
    with Image.open(path) as image:
        return str(imagehash.phash(image.convert("RGB")))


def hamming_distance(left: str, right: str) -> int:
    return bin(int(left, 16) ^ int(right, 16)).count("1")


def stable_json_hash(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(payload).hexdigest()


def labels_to_dict(labels: Iterable[YoloLabel]) -> list[dict[str, object]]:
    return [asdict(label) for label in labels]
