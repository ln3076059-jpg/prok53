# Accuracy-oriented training strategy

The deployable safety detector remains one YOLO11s model with three canonical classes. Accuracy improvements are controlled data, deep-learning, computer-vision, and validation techniques—not extra untracked models.

## Deep learning

- Transfer learning starts from `yolo11s.pt` rather than random initialization.
- MC_001 trains at 960 px with multi-scale batches so small phones and diagonal belt evidence retain more pixels.
- AMP, cosine learning rate, optimizer auto-selection, weight decay, deterministic seed 42, and early stopping are recorded in the bundle.
- Mosaic is reduced by closing it for the last 15 epochs, allowing final fine-tuning on natural cabin composition.
- Validation chooses thresholds per class. Test is frozen and opened only after model and thresholds are locked.

## Computer vision

`training/augment_training_split.py` makes a derived train-only dataset with deterministic low-light, CLAHE, mild motion-blur, sensor-noise, and resolution-loss variants. These transforms keep box geometry unchanged. They are never applied to validation/test, and each derivative retains its parent and effective group.

`training/analyze_cv_quality.py` measures brightness, contrast, Laplacian blur, and per-class object short-side pixels. Flags trigger human inspection; they do not silently delete difficult data.

## Machine-learning/data controls

- The split is group-aware and greedily balances class instances as well as sample counts. Vehicle, clip, subject, and augmentation relatives remain together.
- Rare-class train samples receive at most two deterministic camera-condition variants.
- Hard examples to curate include hands without phones, cups, dashboard screens, bag straps, clothing seams, partial/occluded torsos, night glare, pedestrians outside vehicles, mounted/static phones, and multiple occupants.
- Passenger phones remain positive physical-phone detector examples, not detector negatives. The role-aware event layer makes them `NO_EVENT`.
- Pose/hand/face geometry may later supply `HANDHELD`, `MOUNTED_OR_STATIC`, or `UNKNOWN` context, but it cannot replace physical-phone evidence and must be validated independently before activation.

## Ablation record before the canonical run

Use short, explicitly marked pilot runs only to verify feasibility; do not report them as final results. Compare 640 versus 960 input, built-in augmentation versus the train-only camera derivative, and class/group distribution. Select MC_001 settings from validation only, delete no evidence, then run the canonical Kaggle workflow once. The report keeps detector metrics separate from end-to-end event metrics.
