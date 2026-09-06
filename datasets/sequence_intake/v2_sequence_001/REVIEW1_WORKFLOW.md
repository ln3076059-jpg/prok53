# Temporal development: Codex Review1 and human final confirmation

This workflow applies only to `TEMPORAL_DEVELOPMENT`. It does not change the
independent human ground-truth requirements for untouched holdout, the canonical
sequence validator, identity freezer, evaluator, or existing governance locks.

The operator reviews and edits proposals instead of annotating from zero:

`real video -> technical checks -> Codex REVIEW1_PROPOSAL -> human final decision
-> canonical validation -> temporal ground truth -> calibration when all gates pass`

## Local package

The package lives in `datasets/incoming/v2_sequence_001/review1/` (gitignored).
It contains `rights_review1.csv`, `physical_lineage_review1.csv`,
`sequence_review1.csv`, `identity_roster_review1.json`, per-clip JSON records in
`annotations/`, and `HUMAN_APPROVAL_QUEUE.csv`. Paths resolve from the repository
root. Video bytes and previous `human_review_return/` files are not overwritten.

The current four-clip package is **BLOCKED_FULL_VIDEO_REVIEW_REQUIRED**. Its AI
review examined all 1,781 consecutive decoded frames on exhaustive contact pages,
plus enlarged consecutive transition frames. Native continuous video playback and
audio were not observed. This does **not** fulfill the user's explicit requirement
to review the original video beyond contact sheets. Do not relabel the package
`REVIEW1_READY` based only on extraction or record consistency passing.

The proposals can still be inspected and corrected. All uncertain latch, phone
onset, role and inside/outside boundaries remain explicit. Review1 confidence is
subjective annotation confidence, not a model probability or verified truth.

## Provenance contract

New proposals use:

```text
reviewer_type=AI
reviewer_origin=AI
reviewer_agent=CODEX
review_stage=REVIEW1
status=REVIEW1_PROPOSAL
requires_human_confirmation=true
adjudication_status=PENDING
human_approved=false
```

The new schema is `datasets/schemas/v2_review1.schema.json`; the read-only checker
is `training/validate_review1.py`. It rejects mixed AI/human provenance, stale
video hashes, malformed intervals, gaps/overlaps and frame/time inconsistencies.
All Review1 intervals are zero-based and half-open `[start_frame, end_frame)`;
the final boundary can equal `frame_count`. Times are video-relative seconds,
not real capture timestamps.

Each sequence payload includes a `canonical_annotation_proposal`. Source/camera
and actual capture times stay null; occupant/cabin/vehicle IDs remain proposals.
It is **not canonical GT** and cannot pass the existing canonical validator.
Human identity confirmation and real missing metadata must precede canonical
conversion. Do not manufacture an epoch/date or canonical ID to pass validation.

Historical sparse AI proposals are retained as historical records. They are not
silently migrated or represented as full-video Review1.

## Human final decision

Open `HUMAN_APPROVAL_QUEUE.csv`, starting with C01, then C07, C10 and C11. Each row
links the original video and exact JSON record/payload, exposes the proposed item,
and leaves all human fields blank. Choose one decision for each item:

- `APPROVE`: accept the proposed item after reviewing it.
- `EDIT_AND_APPROVE`: put the corrected item in `EDITED_ITEM_JSON` and explain it.
- `REJECT`: exclude that item; it is not approved truth.
- `UNCERTAIN`: retain uncertainty; it is not approved truth.

The responsible person supplies `REVIEWER_ID`, timezone-aware `REVIEWED_AT`,
`HUMAN_NOTES`, `REVIEW_EVIDENCE_PATH` and `REVIEW_EVIDENCE_SHA256`. Do not use AI
frame sheets as human evidence. Human decisions must bind the actual reviewed
media and final edited payload. A record cannot be approved as a whole while
required items remain unresolved or rejected. Record and payload hashes identify
the draft that was reviewed; they must not be reused for a changed draft.

The optional final-record consistency format requires a human-supplied JSON
receipt with `record_type`, `video_sha256`, `payload_sha256`, `reviewer_id`,
`reviewed_at`, `decision`, and meaningful `review_notes`. `payload_sha256` hashes
UTF-8 JSON with sorted keys, compact separators, unescaped Unicode and no NaN.
The receipt's path/SHA belongs in the final record's `review_evidence` object.
Its decision must be APPROVE or EDIT_AND_APPROVE. A final record uses HUMAN /
FINAL_REVIEW / HUMAN_APPROVED / FINAL, agent null and human_approved true.
Codex must not author these human fields or receipts on the operator's behalf.

The checker establishes document consistency only. A plausible reviewer string,
timestamp and matching hash cannot prove a real human performed the review or
that rights/physical identity are valid. The responsible operator must verify
actual provenance and the substance of the supplied evidence before any promotion.
No conversion or promotion command is provided by this checker.

Rights still require a project-use decision and evidence tied to the media;
source-page URLs are research references. Physical lineage stays NOT_PROVABLE
without documentary source/camera/session/person/vehicle evidence. Candidate IDs,
appearance or different uploaders cannot establish physical independence.

## Consistency command

From repository root, this read-only command checks one Review1 record:

```powershell
py -m training.validate_review1 datasets/incoming/v2_sequence_001/review1/annotations/C01_sequence_review1.json
```

Expected output for a structurally consistent proposal is
`RECORD_CONSISTENCY_PASS`, with `governance_promotion=false`. This is not a
full-video completion check, human approval, intake gate, or canonical GT check.
After real human input, check the returned evidence and edited records first;
then resolve identity/rights/lineage and run the existing canonical validation.

Only Review1 unit tests may be run for this implementation:

```powershell
py -m unittest discover -s tests -p test_review1_contract.py -q
```

Their ephemeral test fixtures are not dataset inputs or real human evidence.
They do not exercise any frozen model test.

## Stages still prohibited

Do not run inference, temporal calibration, external/truth freeze, final event
evaluation or the canonical frozen model test as part of this work. Keep all 12
development clips out of untouched holdout, including copies, renamed clips,
transcodes and derivatives. No additional media download is needed here.

```text
TEMPORAL_CALIBRATION_STATUS=BLOCKED_PENDING_HUMAN_FINAL_CONFIRMATION
TEMPORAL_POLICY_LOCK=PENDING_SEQUENCE_GROUND_TRUTH
EVENT_EVALUATION=PENDING_NEW_UNTOUCHED_HOLDOUT
HUMAN_VERIFIED=false
PRODUCTION_READY=false
FROZEN_TEST_RUN_COUNT=1
```
