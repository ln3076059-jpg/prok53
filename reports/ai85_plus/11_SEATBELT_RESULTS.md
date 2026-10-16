# Seatbelt inference and uncertainty

Fresh inference on the same 641 consumed validation crops: original 224px, 320px, and average of original/horizontal flip at 224px. Each variant has a prediction CSV, per-class P/R/F1, raw and policy confusion matrices, ECE, false-unfastened and abstention in [seatbelt_results.json](seatbelt_results.json).
Labels are model-assisted pending approval. They are not independent confirmed-GT performance. Original CSV replay is included separately because current inference reproduction can differ from stored probabilities.

| Variant | Raw Macro-F1 | Raw balanced acc | Raw FASTENED R | Raw UNFASTENED R | Raw UNCERTAIN R | Policy false-unfastened | Policy abstention |
| --- | --- | --- | --- | --- | --- | --- | --- |
| S0_original_224 | 56.80% | 56.75% | 85.43% | 80.83% | 4.00% | 9.14% | 6.86% |
| S1_resolution_320 | 51.95% | 52.99% | 81.14% | 77.82% | 0.00% | 15.14% | 4.37% |
| S2_horizontal_flip_TTA_224 | 58.52% | 58.40% | 88.86% | 82.33% | 4.00% | 8.00% | 5.46% |

False-unfastened and abstention columns use the inherited 0.50/0.60 operating policy; raw recall/Macro-F1 columns use argmax. Both complete metric sets are in JSON. No thresholds were tuned here. Horizontal flips are exploratory for asymmetric cabin imagery and are not selected for production.
Images are existing torso crops; original-frame localization quality and temporal smoothing cannot be established from unordered crop validation. Occlusion/dark clothing provenance is incomplete. Invisible belt remains uncertain, never automatic unfastened. Additional attention/classifier training remains blocked by confirmed-label and independence prerequisites.
