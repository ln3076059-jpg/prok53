# Seatbelt dataset handoff

Roboflow `seatbelttraining` v4 is downloaded and immutable at `datasets/raw/roboflow_seatbelttraining_v4/4`. Its download tree SHA-256 is `f93a14d1eb5f819685ef873dfd3102b70942cc82e1f6628271a9966a86095f79`.

## Current artifacts

- Ingest manifest: `datasets/manifests/ingest_roboflow_seatbelttraining_v4.jsonl`
- Full review queue: `datasets/manifests/review_queue_roboflow_seatbelttraining_v4.json`
- Source mapping: `mappings/roboflow_seatbelttraining_v4.yaml`
- Audit: `reports/roboflow_seatbelttraining_v4/data_quality.md`

The queue has 8,371 images and 8,451 state proposals: 6,646 `seatbelt_fastened` and 1,805 `seatbelt_unfastened`. None is approved automatically. The source's published splits are unusable because at least 147 inferred source groups cross split boundaries.

## Human review

Start or resume review with a dedicated append-only decision log:

```powershell
py tools/annotation_reviewer/app.py `
  --manifest datasets/manifests/review_queue_roboflow_seatbelttraining_v4.json `
  --decisions datasets/manifests/review_decisions_roboflow_seatbelttraining_v4.jsonl `
  --pending-only
```

For every approved box:

1. Confirm the person is visibly inside a vehicle/cabin and assign `vehicle_context_id`.
2. Assign the sample focus and every box to `driver`, `front_passenger`, `rear_left`, `rear_center`, `rear_right`, or `other_occupant`.
3. Keep comparable upper-body/person geometry for both seatbelt states.
4. Approve `seatbelt_unfastened` only when absence is positively visible. Occlusion, glare, darkness, or an insufficient crop is `UNCERTAIN`.
5. Rebox or delete incorrect proposals. Never turn a thin belt-only box into final state ground truth.

## Materialize reviewed data

After decisions exist:

```powershell
py -m training.apply_review_decisions `
  datasets/manifests/ingest_roboflow_seatbelttraining_v4.jsonl `
  --decisions datasets/manifests/review_decisions_roboflow_seatbelttraining_v4.jsonl `
  --output datasets/reviewed/roboflow_seatbelttraining_v4 `
  --manifest datasets/manifests/reviewed_roboflow_seatbelttraining_v4.jsonl
```

The materializer revalidates vehicle and box-level occupant roles. Final split, near-duplicate audit, test freeze, augmentation, and three-class Kaggle bundling must run only after the approved phone and seatbelt manifests are combined.
