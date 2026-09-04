import os
import shutil
from pathlib import Path

# CPU ONLY INFERENCE - 30 CPU Cores on Kaggle
print("Installing ultralytics...")
os.system("pip install -q ultralytics")

print("Importing ultralytics...")
from ultralytics import YOLO

def main():
    print("Setting up datasets...")
    # Find dataset root dynamically to avoid Kaggle zip extraction path issues
    res = os.popen("find /kaggle/input -name phone_detector -type d | head -n 1").read().strip()
    if res:
        dataset_root = str(Path(res).parent)
        print(f"Found dataset root: {dataset_root}")
        os.system(f"cp -r {dataset_root} /kaggle/working/eval_ds")
    else:
        print("CRITICAL ERROR: Could not find phone_detector in /kaggle/input")
        os.system("find /kaggle/input")
        return
    
    # YOLO classification val() expects a 'train' folder even if it's not used
    print("Faking train folder for classifier...")
    os.system("cp -r /kaggle/working/eval_ds/seatbelt_classifier/val /kaggle/working/eval_ds/seatbelt_classifier/train")

    phone_yaml = Path("/kaggle/working/eval_ds/phone_detector/data.yaml")
    seatbelt_yaml = Path("/kaggle/working/eval_ds/seatbelt_detector/data.yaml")
    cls_data = Path("/kaggle/working/eval_ds/seatbelt_classifier")

    print(f"Phone yaml: {phone_yaml}")
    print(f"Seatbelt yaml: {seatbelt_yaml}")
    print(f"Cls data: {cls_data}")
    
    if phone_yaml.exists():
        phone_yaml.write_text("""path: /kaggle/working/eval_ds/phone_detector
train: images/val
val: images/val
test: images/test
names:
  0: phone
""")
        
    if seatbelt_yaml.exists():
        seatbelt_yaml.write_text("""path: /kaggle/working/eval_ds/seatbelt_detector
train: images/val
val: images/val
test: images/test
names:
  0: occupant_upper_body
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
    if not (Path("/kaggle/working/eval_ds/phone_detector/images/val").exists()):
        print("CRITICAL ERROR: Dataset not copied correctly!")
        os.system("ls -la /kaggle/working/eval_ds")
        return

    try:
        print("Running Phone Detector Validation on CPU...")
        model_phone = YOLO(str(phone_pt))
        model_phone.val(data=str(phone_yaml), split="val", project="/kaggle/working/reports/calibration", name="phone_val", device="cpu", workers=4)

        print("Running Seatbelt Detector Validation on CPU...")
        model_seatbelt = YOLO(str(seatbelt_pt))
        model_seatbelt.val(data=str(seatbelt_yaml), split="val", project="/kaggle/working/reports/calibration", name="seatbelt_val", device="cpu", workers=4)

        print("Running Seatbelt Classifier Validation on CPU...")
        model_cls = YOLO(str(cls_pt))
        model_cls.val(data=str(cls_data), split="val", project="/kaggle/working/reports/calibration", name="classifier_val", device="cpu", workers=4)

        print("Running Phone Detector Test on CPU...")
        model_phone.val(data=str(phone_yaml), split="test", project="/kaggle/working/reports/frozen_test", name="phone_test", device="cpu", workers=4)

        print("Running Seatbelt Detector Test on CPU...")
        model_seatbelt.val(data=str(seatbelt_yaml), split="test", project="/kaggle/working/reports/frozen_test", name="seatbelt_test", device="cpu", workers=4)

        print("Running Seatbelt Classifier Test on CPU...")
        model_cls.val(data=str(cls_data), split="test", project="/kaggle/working/reports/frozen_test", name="classifier_test", device="cpu", workers=4)
    
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
