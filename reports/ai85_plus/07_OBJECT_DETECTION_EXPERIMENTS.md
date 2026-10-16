# Detector validation limits

New governed detector AP50/AP50–95 and per-class P/R: **BLOCKED — no approved bbox validation ground truth established**. DMD action intervals are not phone boxes. The phone bootstrap explicitly restricts its proposals from being reported as final GT. Pending-approval detector labels cannot certify the requested lawful-GT 85% targets.
Embedded checkpoint metrics are preserved in checkpoint_metadata.json as historical training metadata only. A phone AP50 entry in that metadata cannot establish event F1 or roadside performance. The seatbelt detector localizes upper bodies; it is not a belt-state detector.
The installed 8.3.187 validator sets unspecified non-OBB validation confidence to 0.001 (OBB: 0.01), confirmed from its installed source in validation_defaults.json. conf=0.25 was not used as an AP default. NMS IoU=0.70 in spatial probes is distinct from AP IoU=0.50 or the 0.50–0.95 sweep.
Measured spatial diagnostics: seven configurations × thirteen FN midpoints = 91 predictions, with original-frame boxes reconstructed and clamped after crop inference. Static driver ROI containment is not verified occupant-phone association. Torso/hand/face localization improvements require validated localizations and were not claimed from static crops.

| Variant | Frames | Frames with proposals | Proposals | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- |
| A0_full_640 | 13 | 1 | 1 | 237.49 | 1103.71 |
| A1_full_640_TTA | 13 | 1 | 1 | 612.18 | 2024.57 |
| A2_full_960 | 13 | 0 | 0 | 469.46 | 672.68 |
| A2_full_1024 | 13 | 0 | 0 | 500.72 | 726.28 |
| A3_full_640_agnostic | 13 | 1 | 1 | 241.53 | 423.1 |
| A4_driver_640 | 13 | 2 | 2 | 289.88 | 683.18 |
| A4_padded_driver_640 | 13 | 1 | 1 | 255.14 | 587.32 |
