# Kaggle training

## Ready now: MC_BOOTSTRAP_001

Upload `kaggle/mc_bootstrap_001_kaggle_bundle.zip` as a private Kaggle dataset and run `notebooks/kaggle_train_mc_bootstrap.ipynb` on a GPU. Expected output: `/kaggle/working/MC_BOOTSTRAP_001_RECOVERY.zip`.

- Bundle SHA-256: `50b4356124081e5c50404a36d6dac7ecfc2f7ab65b52614c468a48c363f19e9a`
- Images: 14,581 train / 1,782 validation / 1,736 isolated test
- Test instances: 225 phone / 322 fastened / 302 unfastened
- Audit: 118,786 pHash-near pairs grouped with zero cross-split near pairs
- Status: `PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA`

The runner performs full hash and image/label preflight before it requests a GPU. It trains YOLO11s for 150 epochs with transfer learning, 960-pixel multi-scale inputs, AMP, AdamW, cosine decay, mild geometry/color augmentation, and a fixed seed. It writes one Ultralytics checkpoint per epoch (`epoch0.pt` is completed epoch 1) and atomically refreshes `/kaggle/working/MC_BOOTSTRAP_001_LATEST_RESUME.zip` after every epoch.

For recovery in a new session, periodically download that resume ZIP, upload it as a private Kaggle dataset, and attach it alongside the original unchanged bundle. The runner accepts either the intact ZIP or Kaggle's automatically extracted resume directory, then verifies the experiment, dataset/config manifests, checkpoint hashes, and epoch before restoring optimizer state and continuing. A file left only in `/kaggle/working` cannot survive every kind of Kaggle session loss. Report per-class precision, recall, F1, AP50, and AP50-95 from the isolated test split, but label them bootstrap metrics—not final system accuracy.

## Phone proposal bootstrap

The small-object phone bootstrap is ready as `kaggle/phone_bootstrap_001_kaggle_bundle.zip`. Train it with `notebooks/kaggle_train_phone_bootstrap.ipynb`, download `PHONE_BOOTSTRAP_001_RECOVERY.zip`, then run:

```powershell
py -m training.prepare_phone_review_after_kaggle PHONE_BOOTSTRAP_001_RECOVERY.zip --device cpu
```

This model is never final ground truth. It only proposes tight physical-phone boxes on the 334 Mendeley cellphone-behavior frames. A human must correct boxes, confirm a vehicle/cabin context, and assign the occupant role. Passenger phone detections are retained for detector learning but suppressed by the driver-only event rule.

## Canonical three-class model

Build the bundle only after governance, audit, and test freeze:

```bash
python -m training.build_kaggle_bundle --dataset datasets/derived/mc3_v1_aug
```

Upload the resulting zip as a private Kaggle dataset and run `notebooks/kaggle_train_mc001.ipynb` once on a GPU runtime. The bundled runner discovers the dataset path and stops on hash, GPU, class order/count, corruption, isolation, or test-freeze failure. It strips non-Ultralytics metadata from the YAML, uses an absolute runtime data path, inventories the environment, and packages checkpoints plus hashes into `MC_001_RECOVERY.zip`. Do not upload secrets or publish third-party imagery without redistribution permission. See `kaggle/README.md` for the web and CLI handoff.
