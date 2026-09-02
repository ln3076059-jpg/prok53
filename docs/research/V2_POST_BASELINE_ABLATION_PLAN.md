# Post-Baseline Ablation Plan

## A0: Existing V2 Baseline
- **Rationale**: Establish a solid, reproducible control metric using YOLO11s and ByteTrack.
- **Metric**: mAP50-95, Event-level F1.
- **Hypothesis**: The baseline provides sufficient accuracy for real-time edge processing without complex architectures.
- **Control**: V2 dataset with basic augmentation.

## A1: Stronger Data Filtering (Domain Adversarial)
- **Rationale**: Based on Celona et al. and Zhao et al., cross-domain accuracy drops.
- **Metric**: External Test Set mAP.
- **Hypothesis**: Domain adversarial learning will close the performance gap across unseen vehicles.
- **Control**: Compare to A0 baseline on the frozen external test set.

## A2: Attention Modules (CBAM)
- **Rationale**: Guo et al. (2024) indicates attention helps small phone detection.
- **Metric**: mAP on small phone objects.
- **Hypothesis**: Adding CBAM to YOLO11s improves small-object recall.
- **Control**: Compare to A0 baseline.
