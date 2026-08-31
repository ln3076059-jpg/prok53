from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("weights", type=Path)
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    if not Path("reports/model_lock.json").exists():
        raise ValueError("export only after model lock")
    from ultralytics import YOLO

    exported = YOLO(str(args.weights)).export(format="onnx", imgsz=args.imgsz, dynamic=False, simplify=True)
    print(f"Exported {exported}. Run PT versus ONNX smoke comparison before enabling runtime ONNX.")


if __name__ == "__main__":
    main()

