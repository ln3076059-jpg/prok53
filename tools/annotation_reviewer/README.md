# Annotation reviewer

Generate `datasets/manifests/review_queue.json` with `training.build_review_queue`, then run:

```bash
python -m tools.annotation_reviewer.app
```

Open `http://127.0.0.1:8010`. Decisions append to
`datasets/manifests/review_decisions.jsonl`; this audit trail is not overwritten. The UI displays
the attached model proposal, confidence tier, reasons, checks, and model identity before the human
form. Human approval requires a named reviewer, video/vehicle/person/camera identifiers,
conditions, vehicle context, and resolved sample- and box-level occupant roles.

Review 1 records use `reviewer_id=review1`, `reviewer_type=AI`, explicit `admin` delegation and
`REVIEW1_*` statuses. They can never directly emit `APPROVED` or `APPROVED_NEGATIVE`. The Review 1
filters and batch selector are provenance-aware. Admin confirmation requires the literal
`CONFIRM_REVIEW1_PROPOSALS_AS_ADMIN`; after server-side SHA, metadata, condition and role checks it
appends a separate `reviewer_type=HUMAN`, `reviewer_id=admin` record. The UI never triggers that
action automatically.

Batch acknowledgement is limited to visible `HIGH_CONFIDENCE` model-assisted `PASS` proposals and at most 500
samples. It appends to a separate confirmations JSONL with status
`ADMIN_ACKNOWLEDGED_MODEL_PROPOSAL_BATCH` and `governance_eligible: false`; it never fabricates
per-sample `HUMAN_APPROVED` decisions.

A passenger phone remains a valid physical-phone annotation because runtime role logic, not
detector-label falsification, suppresses that event. An unfastened seatbelt remains a violation
for any occupant.

To review a dedicated queue without mixing its audit trail with other reviews:

```powershell
py -m tools.annotation_reviewer.app `
  --manifest datasets/manifests/review_queue_mendeley_phone_bootstrap_001.json `
  --decisions datasets/manifests/review_decisions_mendeley_phone_bootstrap_001.jsonl `
  --acknowledgements datasets/manifests/review_confirmations_mendeley_phone_bootstrap_001.jsonl
```

`GET /api/status` reports the latest decision per sample while the JSONL file retains every
revision. Add `--pending-only` when resuming a large queue. The reviewer hides every sample that
already has an append-only decision; omit the flag when intentionally revising earlier work.

Decision records use schema versions 2–3. Each append records the previous/new status, a required
evidence reason, typed reviewer identity, UTC timestamp, queue/source hashes, stable box IDs and a
decision hash. Revising or confirming a sample creates another line; it never rewrites the earlier
decision.
