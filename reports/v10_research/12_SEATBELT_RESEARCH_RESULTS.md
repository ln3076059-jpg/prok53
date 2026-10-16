# V10 seatbelt research results

| Frozen variant | Macro-F1 | Balanced accuracy | UNCERTAIN recall | False-unfastened | Abstention |
| --- | --- | --- | --- | --- | --- |
| S0_original_224 | 56.80% | 56.75% | 4.00% | 9.14% | 6.86% |
| S1_resolution_320 | 51.95% | 52.99% | 0.00% | 15.14% | 4.37% |
| S2_horizontal_flip_TTA_224 | 58.52% | 58.40% | 4.00% | 8.00% | 5.46% |

Horizontal-flip TTA is the best measured development variant at 58.52%;
it remains far below the 85% goal, and UNCERTAIN recall remains 4%. Resolution 320 degrades Macro-F1 and raises
false-unfastened rate, so increased resolution is rejected as a general remedy.

The pending review manifest contains 500 samples, all PENDING, with zero automatic approved
training labels. Legacy UNCERTAIN errors have no local ROI evidence. Fine-tuning, Triplet Attention, torso-ROI
training, calibration error, selective risk, and coverage curves are therefore **BLOCKED/NOT MEASURED**.
Invisible or occluded belts remain UNCERTAIN and are never relabeled UNFASTENED.
