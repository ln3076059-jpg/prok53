# V9 phone FN/FP root causes

| Case | Action | s | Box frames | Coverage | T0 max | T1 max | Bottleneck |
| --- | --- | --- | --- | --- | --- | --- | --- |
| gB-9-phone-01 | phonecall_right | 29.84 | 0 | 0.00% | 0.423 | 0.438 | PHONE_DETECTOR_NO_PROPOSAL |
| gB-9-phone-03 | phonecall_right | 36.12 | 0 | 0.00% | 0.722 | 0.754 | PHONE_DETECTOR_NO_PROPOSAL |
| gB-9-phone-05 | phonecall_left | 46.61 | 0 | 0.00% | 0.631 | 0.724 | PHONE_DETECTOR_NO_PROPOSAL |
| gB-9-phone-06 | texting_left | 1.01 | 0 | 0.00% | 0.037 | 0.138 | PHONE_DETECTOR_NO_PROPOSAL |
| gB-9-phone-07 | texting_left | 16.53 | 0 | 0.00% | 0.467 | 0.553 | PHONE_DETECTOR_NO_PROPOSAL |
| gB-9-phone-08 | phonecall_left | 39.01 | 2 | 40.00% | 0.676 | 0.686 | PHONE_BOX_PRESENT_ACTION_UNDER_GATE |
| gB-9-phone-09 | texting_left | 12.37 | 5 | 100.00% | 0.417 | 0.452 | PHONE_BOX_PRESENT_ACTION_UNDER_GATE |
| gZ-36-phone-01 | texting_right | 0.81 | 0 | 0.00% | 0.547 | 0.745 | PHONE_DETECTOR_NO_PROPOSAL |
| gZ-36-phone-03 | texting_right | 0.60 | 0 | 0.00% | 0.856 | 0.885 | PHONE_DETECTOR_NO_PROPOSAL |
| gZ-36-phone-05 | texting_right | 0.91 | 0 | 0.00% | 0.916 | 0.935 | PHONE_DETECTOR_NO_PROPOSAL |
| gC-14-phone-02 | texting_right | 12.63 | 0 | 0.00% | 0.771 | 0.842 | PHONE_DETECTOR_NO_PROPOSAL |
| gC-14-phone-05 | texting_left | 0.40 | 0 | 0.00% | — | 0.484 | PHONE_DETECTOR_NO_PROPOSAL |
| gC-14-phone-06 | phonecall_left | 43.65 | 0 | 0.00% | 0.854 | 0.862 | PHONE_DETECTOR_NO_PROPOSAL |

Only 2/13 FN cases have any proposal in the five-frame trace;
0/13 have a verified occupant ID. Dense sampling crosses the fixed 0.80 action gate for
4/13 but matches only
1 event. Geometry cannot rescue a missing box.

Pose landmarks are available for 13/13 cases. Geometry is blocked by a missing phone box
for 11/13 and remains unverified without an occupant track for 2/13;
only 1 proposal has hand/face proximity consistent with handheld use.

The visual audit contains clear large phones at the ear, partial occlusion, ROI-edge placement, glare and
motion blur. Detector domain/representation failure dominates sampling density. The generated
`phone_fn_human_review_queue.csv` preserves every case for human adjudication.


Human adjudication later confirmed phone visibility and use in all 13 cases. Reviewers selected a valid
proposal for 12/13 cases; all selected boxes came from the auxiliary COCO
phone model. `gB-9-phone-06` remained a visible-phone hard example with no valid proposal and was not
written as an empty-label negative. These FN-selected results measure review yield, not detector recall.
