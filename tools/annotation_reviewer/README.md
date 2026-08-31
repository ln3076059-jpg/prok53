# Annotation reviewer

Generate `datasets/manifests/review_queue.json` with `training.build_review_queue`, then run:

```bash
python tools/annotation_reviewer/app.py
```

Open `http://127.0.0.1:8010`. Decisions append to `datasets/manifests/review_decisions.jsonl`; this audit trail is not overwritten. A human must make each decision. Approval requires a vehicle/cabin context ID plus resolved sample- and box-level occupant roles. A passenger phone remains a valid physical-phone annotation because runtime role logic—not detector-label falsification—suppresses that event. An unfastened seatbelt remains a violation for any occupant.

To review the Mendeley queue created after the phone-bootstrap Kaggle run without mixing its audit trail with other reviews:

```powershell
py tools/annotation_reviewer/app.py `
  --manifest datasets/manifests/review_queue_mendeley_phone_bootstrap_001.json `
  --decisions datasets/manifests/review_decisions_mendeley_phone_bootstrap_001.jsonl
```

`GET /api/status` reports the latest decision per sample while the JSONL file retains every revision.

Add `--pending-only` when resuming a large queue. The reviewer will hide every sample that already has an append-only decision; omit the flag when intentionally revising earlier work.
