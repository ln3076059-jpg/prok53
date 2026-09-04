import os
import shutil
from pathlib import Path

# Install ultralytics WITHOUT upgrading dependencies (especially PyTorch)
print("Installing ultralytics without touching PyTorch...")
os.system("pip install -q ultralytics --no-deps")

print("Importing ultralytics...")
from ultralytics import YOLO
import torch

def main():
    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")

    print("Setting up datasets...")
    # Copy all input datasets to working so we can modify them
    os.system("cp -r /kaggle/input/dms-v2-eval-datasets/derived /kaggle/working/eval_ds")
    
    phone_yaml = Path("/kaggle/working/eval_ds/phone_bootstrap_v2/data.yaml")
    seatbelt_yaml = Path("/kaggle/working/eval_ds/seatbelt_v2_balanced/data.yaml")
    cls_data = Path("/kaggle/working/eval_ds/seatbelt_classifier_v2")

    print(f"Phone yaml: {phone_yaml}")
    print(f"Seatbelt yaml: {seatbelt_yaml}")
    print(f"Cls data: {cls_data}")
    
    if phone_yaml.exists():
        phone_yaml.write_text("""path: /kaggle/working/eval_ds/phone_bootstrap_v2
train: images/train
val: images/val
test: images/test
names:
  0: phone
""")
        
    if seatbelt_yaml.exists():
        seatbelt_yaml.write_text("""path: /kaggle/working/eval_ds/seatbelt_v2_balanced
train: images/train
val: images/val
test: images/test
names:
  0: unfastened
  1: fastened
""")

    phone_pt = Path("/kaggle/input/datasets/lethunga/dms-v2-baseline-001/phone_detector/best.pt")
    seatbelt_pt = Path("/kaggle/input/datasets/lethunga/dms-v2-baseline-001/seatbelt_detector/best.pt")
    cls_pt = Path("/kaggle/input/datasets/lethunga/dms-v2-baseline-001/seatbelt_classifier/best.pt")

    print(f"Phone pt: {phone_pt}")
    print(f"Seatbelt pt: {seatbelt_pt}")
    print(f"Cls pt: {cls_pt}")
    
    # Check if models exist
    if not phone_pt.exists() or not seatbelt_pt.exists() or not cls_pt.exists():
        print("CRITICAL ERROR: Models not found!")
        print("Searching for them just in case:")
        os.system("find /kaggle/input -name 'best.pt'")
        return

    # Check if datasets exist
    if not (Path("/kaggle/working/eval_ds/phone_bootstrap_v2/images/val").exists()):
        print("CRITICAL ERROR: Dataset not copied correctly!")
        os.system("ls -la /kaggle/working/eval_ds")
        return

    try:
        print("Running Phone Detector Validation...")
        model_phone = YOLO(str(phone_pt))
        model_phone.val(data=str(phone_yaml), split="val", project="/kaggle/working/reports/calibration", name="phone_val")

        print("Running Seatbelt Detector Validation...")
        model_seatbelt = YOLO(str(seatbelt_pt))
        model_seatbelt.val(data=str(seatbelt_yaml), split="val", project="/kaggle/working/reports/calibration", name="seatbelt_val")

        print("Running Seatbelt Classifier Validation...")
        model_cls = YOLO(str(cls_pt))
        model_cls.val(data=str(cls_data), split="val", project="/kaggle/working/reports/calibration", name="classifier_val")

        print("Running Phone Detector Test...")
        model_phone.val(data=str(phone_yaml), split="test", project="/kaggle/working/reports/frozen_test", name="phone_test")

        print("Running Seatbelt Detector Test...")
        model_seatbelt.val(data=str(seatbelt_yaml), split="test", project="/kaggle/working/reports/frozen_test", name="seatbelt_test")

        print("Running Seatbelt Classifier Test...")
        model_cls.val(data=str(cls_data), split="test", project="/kaggle/working/reports/frozen_test", name="classifier_test")
    
    except Exception as e:
        print("Exception during YOLO inference:")
        print(e)
    finally:
        print("Cleaning up datasets before zipping...")
        os.system("rm -rf /kaggle/working/eval_ds")

        print("Zipping results...")
        shutil.make_archive("/kaggle/working/reports_archive", "zip", "/kaggle/working/reports")
        print("Done! Download reports_archive.zip from output.")

if __name__ == "__main__":
    main()
