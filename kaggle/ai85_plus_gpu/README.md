# Roadwatch AI85+ GPU inference

This private Kaggle script reuses the exact active phone checkpoint, verifies its SHA256 and class map, and measures five frozen-inference settings on 64 deterministic validation images from an existing private legacy Roadwatch dataset. It never reads labels or the test split and never trains.

The legacy dataset is used only for a throughput/proposal probe. This is not an independent accuracy evaluation, a replacement V2 training workflow, or a full-system FPS benchmark. No DMD media or consumed holdouts are uploaded or attached.

Credentials belong in the user's Kaggle credential directory or environment, outside this repository. The upload folder is `kaggle/ai85_plus_private_bundle`; its only payload is the ignored phone `.pt` file and checkpoint metadata. Dataset creation is private by default; do not pass `--public`.

From the repository root, after preparing and verifying that exact checkpoint bundle:

```powershell
$bundle = (Resolve-Path kaggle/ai85_plus_private_bundle).Path
py -3.11 -m kaggle datasets create -p $bundle --quiet
py -3.11 -m kaggle kernels push -p kaggle/ai85_plus_gpu --accelerator NvidiaTeslaT4 --timeout 900
py -3.11 -m kaggle kernels status longngoc11/roadwatch-ai85-plus-frozen-gpu-v8
py -3.11 -m kaggle kernels output longngoc11/roadwatch-ai85-plus-frozen-gpu-v8 -p reports/ai85_plus/kaggle_gpu
```

Creating the dataset is a one-time operation. Existing remote versions and measured local outputs must not be overwritten to conceal a failed run. Fine-tuning still requires confirmed labels, lawful rights and a legitimate split; an API token does not supply those prerequisites.
