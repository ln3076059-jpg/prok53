# V11 false-negative root cause

Evidence scope: authenticated review of the same 13 development false negatives. Each machine-readable case includes timestamp/frame range, action, visibility, raw specialist and auxiliary confidence, driver-ROI and original-frame box coordinates, hand/face proximity, the complete frozen action-probability timeline, gate state, missed interval, and review receipt.

| Event | Time (s) | Action | Specialist max | Selected aux. | T1 max | Association | Failure stages |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gB-9-phone-01 | 39.99-69.83 | phonecall_right | None | 0.102 | 0.438 | UNKNOWN | SPECIALIST_DETECTOR_MISS, AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN, ACTION_EVIDENCE_BELOW_GATE |
| gB-9-phone-03 | 115.15-151.28 | phonecall_right | None | 0.287 | 0.754 | UNKNOWN | SPECIALIST_DETECTOR_MISS, AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN, ACTION_EVIDENCE_BELOW_GATE |
| gB-9-phone-05 | 219.39-265.99 | phonecall_left | None | 0.544 | 0.724 | UNKNOWN | SPECIALIST_DETECTOR_MISS, AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN, ACTION_EVIDENCE_BELOW_GATE |
| gB-9-phone-06 | 266.03-267.04 | texting_left | None | NO VALID BOX | 0.138 | UNKNOWN | SPECIALIST_DETECTOR_MISS, AUXILIARY_DETECTOR_NO_VALID_BOX, OCCLUSION_OR_LOCALIZATION_FAILURE, DRIVER_ASSOCIATION_UNKNOWN, ACTION_EVIDENCE_BELOW_GATE |
| gB-9-phone-07 | 284.98-301.51 | texting_left | None | 0.629 | 0.553 | UNKNOWN | SPECIALIST_DETECTOR_MISS, AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN, ACTION_EVIDENCE_BELOW_GATE |
| gB-9-phone-08 | 315.49-354.50 | phonecall_left | 0.3239066004753113 | 0.536 | 0.686 | UNKNOWN | AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN, ACTION_EVIDENCE_BELOW_GATE |
| gB-9-phone-09 | 370.50-382.86 | texting_left | 0.21571119129657745 | 0.826 | 0.452 | UNKNOWN | AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN, ACTION_EVIDENCE_BELOW_GATE |
| gZ-36-phone-01 | 52.45-53.26 | texting_right | None | 0.393 | 0.745 | UNKNOWN | SPECIALIST_DETECTOR_MISS, AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN, ACTION_EVIDENCE_BELOW_GATE, SUBSECOND_TEMPORAL_RISK |
| gZ-36-phone-03 | 110.72-111.32 | texting_right | None | 0.572 | 0.885 | UNKNOWN | SPECIALIST_DETECTOR_MISS, AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN, SUBSECOND_TEMPORAL_RISK |
| gZ-36-phone-05 | 162.06-162.97 | texting_right | None | 0.027 | 0.935 | UNKNOWN | SPECIALIST_DETECTOR_MISS, AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN, SUBSECOND_TEMPORAL_RISK |
| gC-14-phone-02 | 96.00-108.64 | texting_right | None | 0.240 | 0.842 | UNKNOWN | SPECIALIST_DETECTOR_MISS, AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN |
| gC-14-phone-05 | 225.91-226.31 | texting_left | None | 0.196 | 0.484 | UNKNOWN | SPECIALIST_DETECTOR_MISS, AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN, ACTION_EVIDENCE_BELOW_GATE, SUBSECOND_TEMPORAL_RISK |
| gC-14-phone-06 | 226.34-269.99 | phonecall_left | None | 0.744 | 0.862 | UNKNOWN | SPECIALIST_DETECTOR_MISS, AUXILIARY_CANDIDATE_NOT_ADMITTED_BY_FROZEN_PIPELINE, DRIVER_ASSOCIATION_UNKNOWN |

All 13/13 have a confirmed visible phone and confirmed use. Twelve have a human-selected auxiliary candidate; `gB-9-phone-06` has none. All 13 lack a verified occupant track. Only 4/13 reach the frozen dense action gate, so the dominant path combines specialist detector miss, candidate-admission failure, unknown driver association, and weak action evidence. A reviewed box is proposal yield, not a recovered event; V11 claims zero newly recovered events.
