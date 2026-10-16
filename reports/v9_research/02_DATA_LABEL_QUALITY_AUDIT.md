# V9 data and label quality audit

Scope is development-only: `gB-9, gZ-36, gC-14`. Protected holdouts
`gZ-37, gE-28` were not accessed. Ground truth and event matching
were unchanged.

## Phone evidence

- The same 13 FN IDs occur in the evidence, detector trace and original-video ROI manifest.
- 52 original-video frames were decoded; only
  7 have a phone proposal
  (13.46%).
- No FN has a verified occupant identity. ROI-level association is not an occupant track.
- Visual descriptions are inherited engineering review, not new label approval.

## Seatbelt labels

- Pending review: 500 candidates across 273 groups.
- Review states: `{'PENDING': 500}`; automatic training labels:
  0; occupant roles: `{'PENDING': 500}`.
- Legacy UNCERTAIN queue: 25 samples; locally
  reviewable ROIs: 0.

Fine-tuning is `BLOCKED_PENDING_HUMAN_APPROVAL`. Invisible belt evidence remains
`UNCERTAIN_OR_OCCLUDED`; it is never converted automatically to `UNFASTENED`.
