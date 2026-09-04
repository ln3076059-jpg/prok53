import os
import shutil
import json
from pathlib import Path

def copy_split(src_base, dest_base, split):
    src_images = Path(src_base) / "images" / split
    src_labels = Path(src_base) / "labels" / split
    if src_images.exists():
        shutil.copytree(src_images, Path(dest_base) / "images" / split, dirs_exist_ok=True)
    if src_labels.exists():
        shutil.copytree(src_labels, Path(dest_base) / "labels" / split, dirs_exist_ok=True)

def copy_cls_split(src_base, dest_base, split):
    src = Path(src_base) / split
    if src.exists():
        shutil.copytree(src, Path(dest_base) / split, dirs_exist_ok=True)

def main():
    out_dir = Path("kaggle/eval_datasets/datasets/derived/v2_pretrain_pending_approval")
    # Clean previous eval_datasets
    if Path("kaggle/eval_datasets").exists():
        shutil.rmtree("kaggle/eval_datasets")
    out_dir.mkdir(parents=True, exist_ok=True)

    base = "datasets/derived/v2_pretrain_pending_approval"

    # Phone
    print("Copying phone...")
    phone_src = f"{base}/phone_detector"
    phone_dest = out_dir / "phone_detector"
    copy_split(phone_src, phone_dest, "val")
    copy_split(phone_src, phone_dest, "test")
    shutil.copy2(f"{phone_src}/data.yaml", phone_dest / "data.yaml")

    # Seatbelt
    print("Copying seatbelt...")
    sb_src = f"{base}/seatbelt_detector"
    sb_dest = out_dir / "seatbelt_detector"
    copy_split(sb_src, sb_dest, "val")
    copy_split(sb_src, sb_dest, "test")
    shutil.copy2(f"{sb_src}/data.yaml", sb_dest / "data.yaml")

    # Classifier
    print("Copying classifier...")
    cls_src = f"{base}/seatbelt_classifier"
    cls_dest = out_dir / "seatbelt_classifier"
    copy_cls_split(cls_src, cls_dest, "val")
    copy_cls_split(cls_src, cls_dest, "test")

    # Create dataset metadata
    metadata = {
      "title": "DMS V2 Eval Datasets Canonical",
      "id": "lethunga/dms-v2-eval-datasets-canonical",
      "licenses": [{"name": "CC0-1.0"}]
    }
    Path("kaggle/eval_datasets/dataset-metadata.json").write_text(json.dumps(metadata))

    print("Running kaggle create...")
    os.system("py -m kaggle datasets create -p kaggle/eval_datasets --dir-mode zip")

if __name__ == "__main__":
    main()
