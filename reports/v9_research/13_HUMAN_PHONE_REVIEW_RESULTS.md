# V9 human phone review results

All 13 frozen development false negatives have an authenticated human receipt. Reviewers confirmed
phone visibility and phone use in 13/13 cases. Twelve cases have a selected proposal; one case
(`gB-9-phone-06`) has a visible phone but no valid proposal.

| Case | Action | Selected proposal | Source | Confidence | Materialization |
| --- | --- | --- | --- | --- | --- |
| gB-9-phone-01 | phonecall_right | f1200-C3 | coco_cell_phone | 10.24% | HUMAN_CONFIRMED_BOX |
| gB-9-phone-03 | phonecall_right | f3705-C2 | coco_cell_phone | 28.70% | HUMAN_CONFIRMED_BOX |
| gB-9-phone-05 | phonecall_left | f7560-C1 | coco_cell_phone | 54.35% | HUMAN_CONFIRMED_BOX |
| gB-9-phone-06 | texting_left | NO_VALID_PROPOSAL | — | — | VISIBLE_PHONE_NO_VALID_BOX |
| gB-9-phone-07 | texting_left | f8610-C1 | coco_cell_phone | 62.90% | HUMAN_CONFIRMED_BOX |
| gB-9-phone-08 | phonecall_left | f9390-C1 | coco_cell_phone | 53.58% | HUMAN_CONFIRMED_BOX |
| gB-9-phone-09 | texting_left | f11040-C1 | coco_cell_phone | 82.58% | HUMAN_CONFIRMED_BOX |
| gC-14-phone-02 | texting_right | f3135-C1 | coco_cell_phone | 24.04% | HUMAN_CONFIRMED_BOX |
| gC-14-phone-05 | texting_left | f6735-C1 | coco_cell_phone | 19.58% | HUMAN_CONFIRMED_BOX |
| gC-14-phone-06 | phonecall_left | f7710-C1 | coco_cell_phone | 74.38% | HUMAN_CONFIRMED_BOX |
| gZ-36-phone-01 | texting_right | f1575-C1 | coco_cell_phone | 39.28% | HUMAN_CONFIRMED_BOX |
| gZ-36-phone-03 | texting_right | f3300-C1 | coco_cell_phone | 57.25% | HUMAN_CONFIRMED_BOX |
| gZ-36-phone-05 | texting_right | f4830-C2 | coco_cell_phone | 2.65% | HUMAN_CONFIRMED_BOX |

- Human-reviewed cases: 13/13.
- Valid selected phone proposals: 12/13.
- Selected sources: `{'coco_cell_phone': 12}`.
- Selected confidence range: 2.65%–82.58% (median 46.43%).
- Verified occupant tracks: 0/13.
- Materialized positive diagnostic labels: 12.
- Visible-phone hard examples without labels: 1.
- Automatic training items: 0.

This result establishes that low-confidence auxiliary proposals can surface review candidates for these
FN-selected development cases. It does not estimate detector recall or justify a runtime threshold because
the cases were selected after failure and the chosen boxes came from the proposal pool. The materialized
lane is locked to `DEVELOPMENT_DIAGNOSTIC_ONLY_NOT_TRAINING_READY`; training on it and measuring recovery
on the same three subjects would be leakage. `gB-9-phone-06` has no empty YOLO label, so its visible phone
cannot be learned as background. Protected holdouts remain sealed and production approval remains false.
