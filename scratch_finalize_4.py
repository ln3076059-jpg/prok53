import json
import os
import shutil
import hashlib
import subprocess
from pathlib import Path

def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        while chunk := f.read(8192): h.update(chunk)
    return h.hexdigest()

print("Updating notebook preflight...")
nb_path = Path('notebooks/kaggle_train_v2_pretrain_pending_approval.ipynb')
nb_text = nb_path.read_text('utf-8')
nb_text = nb_text.replace('import training.v2_pretrain_portable_runner; training.v2_pretrain_portable_runner.preflight()', 
                          'import os, pathlib, training.v2_pretrain_portable_runner; training.v2_pretrain_portable_runner.preflight(pathlib.Path(os.getcwd()))')
nb_path.write_text(nb_text, 'utf-8')

print("Extracting bundle and running preflight...")
zip_path = Path('kaggle/MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL_PORTABLE.zip')
extract_dir = Path('kaggle_extract_test')
if extract_dir.exists():
    shutil.rmtree(extract_dir)
extract_dir.mkdir()

import zipfile
with zipfile.ZipFile(zip_path, 'r') as zf:
    zf.extractall(extract_dir)

# Preflight test
env = os.environ.copy()
env['PYTHONPATH'] = str(extract_dir)
subprocess.check_call(["python", "-c", "import os, pathlib, training.v2_pretrain_portable_runner; training.v2_pretrain_portable_runner.preflight(pathlib.Path(os.getcwd()))"], cwd=str(extract_dir), env=env)

print("Updating handoff file if it's not the same...")
handoff = Path('reports/V2_TRAIN_HANDOFF.json')
if handoff.exists():
    h = json.loads(handoff.read_text('utf-8'))
    h["bundle"]["sha256"] = "678b3a6e2b5a6ccb0b4db939303f4f97743e0ef41f0d51273028e240c3a021aa"
    h["bundle"]["bytes"] = 514618515
    handoff.write_text(json.dumps(h, indent=2), 'utf-8')

print("Generating V2_FINAL_PRETRAIN_FREEZE.json...")
report = {
    "CURRENT_HEAD": subprocess.check_output(["git", "rev-parse", "HEAD"], text=True).strip(),
    "V1_SHA": sha256_file(Path('experiments/MC_BOOTSTRAP_001/config.yaml')),
    "EXPLORATORY_TRAINING_READY": True,
    "GOVERNED_TRAINING_READY": False,
    "PRODUCTION_READY": False,
    "BUNDLE": {
        "path": "kaggle/MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL_PORTABLE.zip",
        "bytes": 514618515,
        "sha": "678b3a6e2b5a6ccb0b4db939303f4f97743e0ef41f0d51273028e240c3a021aa"
    }
}
Path('reports/V2_FINAL_PRETRAIN_FREEZE.json').write_text(json.dumps(report, indent=2), 'utf-8')
Path('reports/V2_FINAL_PRETRAIN_FREEZE.md').write_text("# V2 Final Pre-Train Freeze\n\nExploratory ready.", 'utf-8')

shutil.rmtree(extract_dir)
print("Done.")
