# GitHub and local audit

Verified local branch `roadwatch-v4-candidate`, HEAD `967bdebf0bdfea953d84fca8a8b969d223a81d72`. Remote candidate branch agrees with local HEAD; main is older. V7 is the latest available implementation; no V8 artifact was found.

```json
{
  "remote_refs": [
    "925e81a5a76fcf6bcadb32ba9e9633011d12e572\tHEAD",
    "925e81a5a76fcf6bcadb32ba9e9633011d12e572\trefs/heads/main",
    "967bdebf0bdfea953d84fca8a8b969d223a81d72\trefs/heads/roadwatch-v4-candidate"
  ],
  "status_counts": {
    " M": 8,
    " D": 3558,
    "??": 14
  },
  "modified_source_paths": [
    "PRODUCT.md",
    "backend/services/analysis_queue.py",
    "backend/services/uploads.py",
    "reports/v4_upgrade/experiment_preflight.json",
    "tests/test_engineering_freeze.py",
    "tests/test_v4_experiment_runner.py",
    "training/dmd/run_v31_ablation.py",
    "training/dmd/run_v4_experiments.py"
  ],
  "python": "3.11.9",
  "torch": "2.13.0+cpu",
  "cuda_available": false,
  "memory": {
    "total": 6297477120,
    "available": 253583360,
    "percent": 96.0,
    "used": 6043893760,
    "free": 253583360
  },
  "disk": {
    "total": 323389222912,
    "used": 312580788224,
    "free": 10808434688
  }
}
```

Existing edits and deletions were preserved. No reset, clean, checkout, restore, commit, push, production activation or new dataset download was performed. The audit records the working tree at audit time, including the new untracked work. Active weight hashes are in `environment_audit.json`.
