# Thirteen original false negatives

Original-video midpoint evidence and dense temporal traces are linked in the machine artifacts. Visual comments are inherited frame inspections, not human confirmation of every frame or revised labels. Lighting/angle, true phone pixels and hand/face association lack validated annotations and remain unknown; proposal sizes are reported separately.

| Case | Duration s | T0 centers | T1 centers | T1 max raw phone p | T1 matched | Midpoint observation |
| --- | --- | --- | --- | --- | --- | --- |
| gB-9-phone-01 | 29.839 | 59 | 296 | 0.43777573108673096 | False | Phone clearly visible at right ear; hand and device occupy a large region. |
| gB-9-phone-03 | 36.122 | 72 | 358 | 0.7543731927871704 | False | Phone clearly visible at right ear; little occlusion. |
| gB-9-phone-05 | 46.606 | 92 | 462 | 0.7238595485687256 | False | Phone clearly visible at left ear; strong behavior evidence. |
| gB-9-phone-06 | 1.008 | 2 | 11 | 0.13816387951374054 | False | Short interval; gaze and hands move toward lap, but phone is occluded. |
| gB-9-phone-07 | 16.532 | 33 | 165 | 0.5525065660476685 | False | Phone clearly visible in left hand near torso. |
| gB-9-phone-08 | 39.012 | 78 | 387 | 0.6858704090118408 | False | Phone at left ear, partly behind hand but behavior is clear. |
| gB-9-phone-09 | 12.366 | 24 | 122 | 0.45249930024147034 | False | Phone clearly visible in left hand near torso. |
| gZ-36-phone-01 | 0.806 | 1 | 8 | 0.7452247738838196 | False | Phone clearly visible in right hand near lap. |
| gZ-36-phone-03 | 0.605 | 1 | 6 | 0.8854799866676331 | False | Phone clearly visible in right hand near torso. |
| gZ-36-phone-05 | 0.907 | 2 | 9 | 0.9352058172225952 | False | Phone clearly visible in right hand; bright screen and partial glare. |
| gC-14-phone-02 | 12.635 | 25 | 125 | 0.8423033356666565 | False | Phone is small and partly clipped at the lower-right ROI boundary. |
| gC-14-phone-05 | 0.403 | 0 | 3 | 0.48383933305740356 | False | Phone is visible during rapid hand motion with substantial motion blur. |
| gC-14-phone-06 | 43.649 | 86 | 433 | 0.8618807792663574 | True | Phone clearly visible at right ear; little occlusion. |

The 0.403-second gC-14 interval gains dense centers but remains unmatched. Across all four subsecond events, T1 achieves coverage 4/4 and matching 0/4. Temporal coverage is therefore only one bottleneck; raw action evidence, smoothing, minimum support and frozen matcher compatibility still limit detection.
ROI clipping, blur and occlusion are visible in existing examples. Padding was measured and did not improve proposal coverage over the unpadded driver crop in this midpoint diagnostic. No geometry hard gate was added.

[Per-case traces](evidence_diagnostics.json), [original prior ROI manifest](../ai85_upgrade/fn_frame_manifest.json), [new detector boxes](phone_spatial_probes.json).
