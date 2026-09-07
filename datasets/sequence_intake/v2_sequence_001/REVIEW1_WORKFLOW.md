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

The current four-clip package is **REVIEW1_READY_FOR_HUMAN_CONFIRMATION** under
the explicitly revised visual-task contract below. The earlier native-playback
blocker is retained in local history. It was not a missing-frame finding.

`REVIEW1_FULL_VISUAL_PASS` requires inspection of every decoded frame without
sampling, in presentation order from frame 0 through the final frame; original
PTS/timing and video SHA must match; decode must pass; all state transitions must
be inspected at consecutive-frame resolution; visible cuts/discontinuities must
be recorded. Merely generating an exhaustive sheet does not establish inspection.
Sparse contact-sheet sampling cannot pass this contract.

For PHONE/SEATBELT labels, native audio playback is not required unless a proposed
label actually depends on audio. Neither native playback nor listening occurred
here. Codex visually inspected all consecutive frame pages in the preceding turn
and consecutive transition enlargements then and in the continuation. Image
display resized frames; uncertain/occluded details remain explicitly uncertain.

| Clip | Total frames | Inspected frames | Coverage | First / last frame |
|---|---:|---:|---:|---|
| C01 | 454 | 454 | 100% | 0 / 453 |
| C07 | 646 | 646 | 100% | 0 / 645 |
| C10 | 441 | 441 | 100% | 0 / 440 |
| C11 | 240 | 240 | 100% | 0 / 239 |

All four source hashes and original per-frame PTS were reverified through EOF.
No visible shot cuts were observed in the complete ordered visual inspection.
This finding does not establish original capture speed, physical independence,
or visibility of hidden occupants. Local `full_visual_review.json` and per-clip
`Cxx_full_visual_review.json` record source hashes, page/frame ranges, transition
findings, image hashes and per-frame PTS evidence. The inspection count is a Codex
review record; it is not inferred from decoder success or file existence.

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

Each record type now has a typed payload contract:

- Rights: source/terms URL, creator, asset ID, recommendation, reason and evidence
  availability. Unknown source details may be null for a pending decision.
  ACCEPT_CANDIDATE needs all source details and existing nonempty SHA-bound
  evidence. No AI payload may claim human project-use approval.
- Physical lineage: nullable source/camera/session/vehicle/person proposals,
  lineage status and evidence basis. NOT_PROVABLE carries no asserted IDs;
  other statuses need complete proposed IDs and SHA-bound evidence. A structured
  proposal and matching evidence hash still do not prove physical independence.
- Identity: proposed vehicle/cabin IDs and unique occupants, canonical role enum,
  nullable inside-vehicle proposal, evidence basis and confidence.
- Sequence: typed occupants and full PHONE/SEATBELT/context intervals. Interval
  roles must match the referenced occupant. Allowed visibility values are
  `clear`, `partial`, `occluded`, `out_of_view`, `unknown`.

The canonical role enum is `driver`, `front_passenger`, `rear_left`,
`rear_center`, `rear_right`, `unknown`. These are proposals, not physical IDs or
human-approved identities. Nested structured AI claims of human approval fail.

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

Run Review1 unit tests first for changes to this implementation:

```powershell
py -m unittest discover -s tests -p test_review1_contract.py -q
```

Their ephemeral test fixtures are not dataset inputs or real human evidence.
They do not exercise any frozen model test.

Ordinary full `py -m pytest` is also authorized for these executable-code changes.
For clean provenance, first commit code/schema/tests, check the tracked tree is
clean, then run `py tests/run_tests_with_provenance.py` (which runs full pytest).
It records the tested code HEAD, pytest counts and log SHA in
`tests/test_provenance.json`. A later provenance-only commit may contain the
resulting log/metadata. Do not describe dirty-tree tests as testing exact HEAD;
do not rerun pytest just to make the later provenance-only commit the tested SHA.

## Giao diện xác nhận bằng tiếng Việt

Người duyệt có thể dùng [công cụ duyệt video local](../../../tools/temporal_reviewer/README.md):

```powershell
py -m tools.temporal_reviewer.app
```

Mở `http://127.0.0.1:8766` để xem video gốc, duyệt từng mục hoặc phê duyệt tất cả
mục đang chờ trong clip / toàn queue. Mỗi lần lưu cần người thật nhập danh tính,
thời điểm có múi giờ, ghi chú và xác nhận nội dung đã xem. Công cụ ghi biên nhận
cấp item và 7 cột HUMAN; không sửa proposal AI, không ghi đè mục đã có human input.
Biên nhận thao tác không thay thế evidence rights/lineage hoặc final receipt toàn
payload; không tự chuyển canonical và không thay đổi các trạng thái governance.

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
