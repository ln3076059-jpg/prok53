# V12 seatbelt results

The current review manifest contains 500 candidates, all PENDING, with zero automatic training labels. Rights and approved group splits are also absent. V12 therefore did not train, calibrate, or relabel the seatbelt branch.

The inherited development best remains Macro-F1 58.52%. Its source variants report UNCERTAIN recall 4% for the best horizontal-flip result. This does not meet the 85% objectives. `UNCERTAIN_OR_OCCLUDED` remains distinct from `UNFASTENED`; invisible belts were not converted into violations. False-unfastened, abstention and balanced accuracy remain the inherited values in `metric_comparison.json`, with no new independent evidence.

Triplet Attention, visibility/state multitask fitting, torso ROI training and calibration remain BLOCKED pending human-approved three-state labels, explicit fitting rights and a subject-disjoint split.
