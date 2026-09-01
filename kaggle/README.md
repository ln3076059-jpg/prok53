# Kaggle web handoff

## Multi-model V2 portable Windows handoff

The active ready-to-copy bundle is
`kaggle/MULTIMODEL_V2_PRETRAIN_PENDING_APPROVAL_PORTABLE.zip` (514,618,511 bytes), SHA-256
`87c3b4429215f57dd0bb2e7b8625f079bd58ce64f349c9e3c6d26977e0dc8b77`. It contains the phone
detector, upper-body detector, three-state seatbelt classifier, both offline YOLO base weights,
safe 8/16 GB profiles, and automatic resume/recovery code. After extracting it on the Windows
GPU machine, run `.\START_V2_PRETRAIN_REVIEW_TRAINING.ps1 -Mode preflight`, then
`.\START_V2_PRETRAIN_REVIEW_TRAINING.ps1 -InstallDependencies`. The artifact is explicitly
`MODEL_ASSISTED_PENDING_APPROVAL`: it supports exploratory training but does not claim human
verification, governed metrics, or production readiness.

## Train now: three-class proposal bootstrap

The ready-to-upload file is `kaggle/mc_bootstrap_001_kaggle_bundle.zip` (374,903,639 bytes). Its SHA-256 is:

```text
b8f42ad0bcbf02a2a5ec461eb66c29204f49314d0f1715cd4d38655bed42564b
```

The bundle contains 10,018 images (6,500 selected train, 1,782 unchanged validation, 1,736 unchanged test) with the exact class order `phone`, `seatbelt_fastened`, `seatbelt_unfastened`. The reduced train split retains every phone-positive and unfastened-positive image, covers all 3,288 original source groups, and preferentially retains difficult fastened/hard-negative examples. Local audit passes with zero SHA/group/component/near-duplicate cross-split overlap and every split contains all three classes.

Web workflow:

1. In Kaggle, create a new **private** dataset and upload `mc_bootstrap_001_kaggle_bundle.zip`. Do not publish it because the bundle combines third-party sources with different attribution requirements.
2. Create a new notebook and enable a GPU accelerator plus Internet access.
3. Attach the private dataset created in step 1.
4. Import `notebooks/kaggle_train_mc_bootstrap.ipynb` and choose **Run All**.
5. While training, periodically download `/kaggle/working/MC_BOOTSTRAP_001_LATEST_RESUME.zip`. After completion, download `/kaggle/working/MC_BOOTSTRAP_001_RECOVERY.zip`.

The runner stops before training if any file hash, class order, bbox, split count, usage policy, or isolation gate differs. Training uses YOLO11s transfer learning at 960 px for 150 epochs with AMP, AdamW, cosine learning rate, multi-scale training, and controlled camera-condition augmentation. `save_period: 1` retains `epoch0.pt` through `epoch149.pt`; the patience value is 150 so normal early stopping cannot truncate the requested run. After every epoch, an atomic lightweight resume archive stores optimizer-bearing `last.pt`, `best.pt`, `results.csv`, `args.yaml`, hashes, and the completed epoch.

To continue in a fresh Kaggle session, upload the latest resume ZIP as a second **private** Kaggle dataset and attach it alongside the unchanged training bundle. The notebook auto-discovers and validates either the intact ZIP or Kaggle's extracted `resume_metadata.json`/`weights/` layout, then chooses the newest valid epoch. Kaggle cannot recover `/kaggle/working` after a hard session loss unless the resume file was downloaded or committed as notebook output, so save it periodically. The final recovery archive contains the essential best/last checkpoints and reports; historical epoch checkpoints remain in the notebook output directory to avoid duplicating several gigabytes inside the recovery ZIP.

This is a **proposal/demo bootstrap**, not the scientifically final `MC_001`. Passenger phone boxes remain detector labels but phone violations are driver-only at event level; unfastened-seatbelt violations apply to every associated occupant. Human review is still required before final accuracy claims.

## Stage 0: phone proposal bootstrap

The generic COCO detector has very low recall on tiny/occluded in-cabin phones. Build the private, one-class auxiliary bundle:

```powershell
py -m training.build_phone_bootstrap_bundle
```

Upload `kaggle/phone_bootstrap_001_kaggle_bundle.zip` as a private Kaggle dataset and run `notebooks/kaggle_train_phone_bootstrap.ipynb` with a GPU. Download `PHONE_BOOTSTRAP_001_RECOVERY.zip`. Its weights are **proposal-only**: use them to pre-annotate physical phones, then require human approval and driver/passenger association.

After downloading the recovery archive into the project root, verify it, install the immutable proposal weight, run it over all 334 Mendeley `Cellphone Use` frames, and create the focused human-review queue with one command:

```powershell
py -m training.prepare_phone_review_after_kaggle PHONE_BOOTSTRAP_001_RECOVERY.zip --device cpu
```

Use `--device 0` on a local CUDA GPU. The importer rejects the wrong experiment, usage policy, frozen split counts, metrics metadata, unsafe archive paths, or a mismatched `best.pt` SHA-256. It installs the model under `models/proposals/phone_bootstrap_001/` and writes:

- `datasets/manifests/phone_proposals_mendeley_bootstrap_001.json`
- `datasets/manifests/review_queue_mendeley_phone_bootstrap_001.json`

Even when no phone is proposed, the corresponding cellphone-behavior frame stays in the focused queue for manual inspection. Start the reviewer with the command in `tools/annotation_reviewer/README.md`; approval requires vehicle context and a resolved driver/passenger role.

## Stage 1: canonical three-class model

The repository cannot create your Kaggle account slug or accept third-party data licenses for you. After data review and test freeze:

```powershell
py -m training.build_kaggle_bundle --dataset datasets/derived/mc3_v1_aug
```

Create a **private** Kaggle dataset named `mc001-bundle`, upload `mc001_kaggle_train_bundle.zip` (unzipped by Kaggle), attach it to a new GPU notebook, and import `notebooks/kaggle_train_mc001.ipynb`. The notebook discovers the bundle by its hash manifest; no hard-coded account path is required.

For CLI publishing, copy the two `*.example.json` files, replace `YOUR_KAGGLE_USERNAME`, and follow Kaggle's dataset/kernel CLI commands. Do not publish source images unless every source license and attribution permits redistribution.

Expected output: `/kaggle/working/MC_001_RECOVERY.zip`, containing both checkpoints, results, arguments, environment inventory, bundle commit, and the best-weight SHA256.
