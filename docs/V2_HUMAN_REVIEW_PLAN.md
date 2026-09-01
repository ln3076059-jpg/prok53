# V2 Human Review Plan

Status: **BLOCKED_BY_HUMAN_REVIEW**. Model-assisted review is proposal triage, not ground truth.

## Decision contract

Every reviewer action is appended to JSONL. Schema version 2 records `sample_id`, optional stable
`box_id`, `previous_status`, `new_status`, `reviewer_id`, `reviewer_type`, timezone-aware
`reviewed_at_utc`, `decision_reason`, notes, source queue identity/hash, source asset/group, and a
decision hash. Revisions append a new transition; they never edit an earlier line.

Only a named human can create `APPROVED` or `APPROVED_NEGATIVE`. Automated passes remain
`MODEL_ASSISTED_PROPOSAL_PENDING_APPROVAL` and `governance_eligible: false`.

## Exact review order

1. Review the 500 seatbelt uncertainty candidates. Unclear, dark, cropped, reflected or occluded
   belt evidence remains `UNCERTAIN_OR_OCCLUDED`.
2. Review the prioritized phone-negative queue. It contains 3,013 zero-box group representatives;
   model detections, adverse visibility and uncertain cases appear first. Every item remains
   `PENDING` until inspected.
3. Review 1,216 phone-positive group representatives. Rebox the physical phone and resolve vehicle
   context and occupant role; reject product photos or out-of-vehicle examples.
4. Review 4,868 seatbelt upper-body/state proposals. Confirm comparable upper-body ROIs and one of
   fastened, unfastened or uncertain/occluded.
5. Review the 334 Mendeley phone and 2,704 ADT external candidates without admitting them to
   development data.

Start the prioritized negative queue with:

```powershell
py tools/annotation_reviewer/app.py `
  --manifest datasets/manifests/v2_phone_negative_priority_review.json `
  --decisions datasets/manifests/v2_phone_negative_decisions.jsonl `
  --pending-only
```

## Review evidence

Phone review covers physical-phone correctness, missing phone boxes, false positives, ambiguity,
mounted/static context, passenger context and crop quality. Seatbelt review covers upper-body ROI,
all three classifier states, invisible evidence, reflection, bags/straps, clothing folds and
partial occupants.

## Seatbelt hard-negative capture gap

The exact capture manifest is absent, so the review queue cannot yet be populated. Current
coverage is 0 samples for each of: `empty_cabin`, `empty_seat`, `reflection`, `steering_wheel`,
`clothing_folds`, `bag_or_strap`, `seat_pattern`, `non_occupant_upper_body_like`,
`partial_occupant`, and `windshield_glare`. Missing scenarios: **10/10**. Status remains
`BLOCKED_BY_DATA`; no minimum sample count is invented without a capture protocol decision.

## Exit gate

The governed lane remains closed until all admitted samples have append-only human decisions,
complete camera/video/vehicle/person metadata, confirmed conditions, resolved roles, group-clean
splits and a refreshed readiness audit. Administrative batch acknowledgement does not clear it.
