# V10 generalization validation

Status: **DATA_GENERALIZATION_BLOCKED**.

`gB-9`, `gC-14`, and `gZ-36` are repeatedly used development subjects. Protected `gZ-37` and `gE-28` were
not accessed. There are no new approved subjects and no independent holdout. DMD rights remain
`RIGHTS_ACCEPTED=false`, `CANONICAL_ELIGIBLE=false`; external upload and V10 fine-tuning are blocked.

The D2 LOSO folds are subject-disjoint but still LEVEL 2 because the three subjects and hypotheses have been
examined repeatedly. Confidence intervals and an independent claim are inappropriate. A valid next stage needs
provenance-approved subjects, deduplication, group-disjoint train/validation, a locked unseen test, then a
single frozen model/threshold/event-policy evaluation.
