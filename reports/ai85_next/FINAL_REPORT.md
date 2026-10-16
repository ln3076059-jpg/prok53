# AI85 next-stage final report

The work identified subject/camera appearance bias as the main SAFE classification failure and
insufficient continuous action evidence as the main FN mechanism. A new subject-normalized
representation was implemented and evaluated without rereading `gZ-37` or `gE-28` and without
repeating the failed Mean, Attention, or Native112 fine-tunes.

The direct coarse probe improves SAFE recall from
`12.57%` to
`70.81%`. The controlled binary PHONE branch
with the inherited policy reaches Precision 80.00%, Recall 55.17%, F1 65.31%, and 0.1813 FA/min.
Nested policy selection does not improve the baseline. An exploratory policy reaches Precision
88.89% and 0.0906 FA/min but remains at Recall 55.17% and is not eligible for selection.

Five baseline FNs are recovered by the new fixed branch, but five prior matches are lost, leaving
overall recall unchanged. The detector proposes PHONE in only 2/13 FN intervals at 960 input.
One FN lacks any centered action sample and one fails temporal matching after sustained evidence.

## Decision

- `AI85_TARGET_ACHIEVED = false`
- `AI85_INDEPENDENT_VALIDATION = NOT_ESTABLISHED`
- `SELECTED_CANDIDATE = none`
- `PRODUCTION_APPROVED = false`
- Production models and thresholds remain unchanged.

## Remaining blockers

1. No new approved subjects and no untouched licensed holdout.
2. Source rights remain pending project review.
3. BODY-only availability for gC-14 prevents a clean multi-view group-disjoint ablation.
4. No phone-box ground truth or persistent phone/occupant association exists.
5. The current 0.5-second action sampling cannot cover every sub-second event.
6. Subject-normalized probabilities require calibration on separate validation subjects before
   any temporal policy can be frozen.
