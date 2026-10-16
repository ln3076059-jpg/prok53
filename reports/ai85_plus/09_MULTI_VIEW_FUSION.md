# Availability-aware multiview evidence

BODY/FACE/HANDS availability is read from the actual legacy records. Missing views are represented by `None` and excluded from confidence fusion denominators. gC-14 has BODY only; a missing FACE/HANDS camera never contributes negative evidence.

| Subject | BODY available | FACE available | HANDS available |
| --- | --- | --- | --- |
| gB-9 | 809 | 809 | 809 |
| gZ-36 | 1024 | 1024 | 1024 |
| gC-14 | 795 | 0 | 0 |

The new availability-aware helper was exercised on cached evidence; its aggregate maximums are diagnostic confidence summaries, not calibrated probabilities or an integrated event gain. Existing cache lacks aligned phone boxes plus hand/face landmarks and occupant tracks. No normalized ear/phone distances, head pose, MediaPipe integration or reliability calibration was fabricated. No geometry threshold was selected on protected subjects, and occluded phones were not hard-rejected by geometry.
Further multi-task fusion requires quality-checked synchronized views and landmarks. Sources in references.md motivate this direction only.
