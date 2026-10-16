# Seatbelt metric recomputation and training boundary

Recomputed 641 validation predictions from the original CSV, SHA256 `22d57b98ebcfc533ebe88220e692ac21181b74e1f81bf5cf3cf36e06a0d777c5`. Raw macro-F1 57.28%, balanced accuracy 57.22%.

| Class | Support | Precision | Recall | F1 |
| --- | --- | --- | --- | --- |
| seatbelt_fastened | 350 | 83.10% | 85.71% | 84.39% |
| seatbelt_unfastened | 266 | 78.70% | 81.95% | 80.29% |
| uncertain_or_occluded | 25 | 33.33% | 4.00% | 7.14% |

Post-policy false-unfastened rate: 10.29%. Confusion matrices, abstention denominators and ECE are in `seatbelt_metrics.json`.

The available crop dataset contains 3,660 training images but zero UNCERTAIN examples; its report says human review `PENDING` and ready_for_training `False`. New three-state fine-tuning is blocked by this label/provenance gap. The 25 validation UNCERTAIN samples were not reused as training data. No threshold was tuned against this CSV, and frozen test data was not accessed.
