import os
import zipfile
import builtins
from pathlib import Path
from kaggle.api.kaggle_api_extended import KaggleApi

_original_open = builtins.open
def _utf8_open(*args, **kwargs):
    if len(args) > 1 and 'w' in args[1] and 'b' not in args[1]:
        kwargs['encoding'] = 'utf-8'
    return _original_open(*args, **kwargs)
builtins.open = _utf8_open

api = KaggleApi()
api.authenticate()

print("Downloading final output...")
try:
    api.kernels_output("lethunga/dms-v2-baseline-evaluation", "kaggle/final_output")
except Exception as e:
    print(f"Ignored exception: {e}")

print("Extracting reports...")
zip_path = Path("kaggle/final_output/reports_archive.zip")
if zip_path.exists():
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall("kaggle/final_output/reports")
    print("Done extracting to kaggle/final_output/reports")
else:
    print("reports_archive.zip not found! Files downloaded:")
    os.system("dir kaggle\\final_output")
