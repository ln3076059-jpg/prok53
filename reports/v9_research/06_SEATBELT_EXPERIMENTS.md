# V9 seatbelt experiments

| Variant | Macro-F1 | Balanced accuracy | UNCERTAIN recall | False-unfastened | Abstention |
| --- | --- | --- | --- | --- | --- |
| S0_original_224 | 56.80% | 56.75% | 4.00% | 9.14% | 6.86% |
| S1_resolution_320 | 51.95% | 52.99% | 0.00% | 15.14% | 4.37% |
| S2_horizontal_flip_TTA_224 | 58.52% | 58.40% | 4.00% | 8.00% | 5.46% |

Best measured development Macro-F1 is 58.52%. Of 25 legacy
UNCERTAIN failures, 18 are
confidently wrong at ≥0.90. Calibration alone cannot repair the missing class. Torso ROI, Triplet Attention
and hard-negative fine-tuning remain blocked by unapproved labels. No weights were changed.
