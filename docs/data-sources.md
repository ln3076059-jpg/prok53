# Dataset source plan

No source below is automatically part of the final dataset. Download permission, media provenance, semantic review, vehicle context, grouping, duplicate isolation, and human approval are separate gates.

| Priority | Source | License shown by host | Intended use | Mandatory correction |
|---|---|---|---|---|
| P1 | [Kaggle DMS](https://www.kaggle.com/datasets/habbas11/dms-driver-monitoring-system) | Apache-2.0 | Physical-phone proposals, belt-object cues, hard negatives | Never convert bare `Seatbelt` or its absence directly into either seatbelt state; redraw comparable occupant upper-body boxes. |
| P1 | [Mendeley Driver Risk Behavior](https://data.mendeley.com/datasets/562zj8n7xf/1) | CC-BY-4.0 | In-vehicle cellphone behavior, negative gestures, multiple vehicles/lighting | The 138 MB source uses behavior-region YOLO labels. `Cellphone Use` is not a physical-phone box; manually redraw visible phones and review grouping. |
| P2 | [Kaggle real-car seatbelt photos](https://www.kaggle.com/datasets/alexandresintes/seatbelt-detection-dataset-real-car-photos) | CC-BY-4.0 | Fastened-seatbelt candidates | Check the limited person/domain diversity and redraw upper-body ROIs. |
| P2 | [Roboflow seatbelttraining v4](https://universe.roboflow.com/seatbelttraining-7yh0f/seatbelt-detection-lb1ec/dataset/4) | CC-BY-4.0 | Seatbelt candidates | Preserve source/augmentation lineage in one group; review all label semantics. |
| P2 | [Anyway Labs synthetic DMS](https://huggingface.co/datasets/anywaylabs/synthetic-driver-monitoring-detection) | CC-BY-4.0 | Camera-angle robustness and pipeline smoke tests | `calling`/`texting` behavior boxes are not physical-phone boxes; synthetic performance must be reported separately. |
| P3 | [c3rl seatbelt classification](https://huggingface.co/datasets/c3rl/seatbelt-detection) | MIT shown on repository | Synthetic smoke test / separately reported ablation only | A four-image spot audit across both folders found geometric synthetic scenes rather than real cabin occupants. Do not admit it into governed real-camera training; folder classes are not detector boxes. |
| Deferred | [Kaggle Driver Activity](https://www.kaggle.com/datasets/guanhualee/driver-activity-dataset) | CC-BY-SA-4.0 | Optional four-second temporal benchmark | The 16,600,925,334-byte archive is deferred. It is not required for the first model and action labels still need manual physical-phone annotation. |

Rejected sources remain in `datasets/sources.yaml`. In particular, a Roboflow dataset without an explicit license is rejected because [Roboflow documents that unspecified-license datasets remain all-rights-reserved](https://docs.roboflow.com/universe/find-a-dataset-on-universe).

For V2 external evaluation, the 334 Mendeley `Cellphone Use` frames are now isolated in
`datasets/manifests/v2_phone_external_mendeley_review.json`. They span five inferred sequence
groups and remain `EXTERNAL_TEST_ONLY_UNTIL_MODEL_LOCK`; the behavior-region labels are never
treated as phone boxes, and every physical phone/negative requires manual review. The ADT queue
uses the same frozen-candidate policy for seatbelt/phone evaluation.

## Seatbelt-source decision, 2026-08-31

Roboflow `seatbelttraining` v4 is now downloaded with an immutable hash manifest. Its host/source data reports CC BY 4.0 and 8,371 generated images (7,323/696/352), including three augmented outputs per training example. The actual data contains `no-seatbelt` and `seatbelt` person/upper-body boxes, but at least 147 inferred groups leak across its published splits. Those splits are discarded; augmentation-lineage regrouping and human state/ROI review remain mandatory.

DriverMVT was evaluated but deferred because Part 1 alone is 39.7 GB. The revised DMD corpus was also rejected for this project handoff: its official page describes roughly 25 TB raw and limits use to academic purposes under CC BY-NC-ND 4.0. Neither is a practical lightweight replacement for the already downloaded 138 MB Mendeley Driver Risk source.

## V2 gap-source decision, 2026-09-01

ADT `Seat_belt_detection` v1 is downloaded under `datasets/raw/roboflow_adt_seatbelt_v1/1`.
The source contains 3,820 images but mixes twelve classes. Audit found 2,704 inferred
augmentation-lineage groups, including 56 `mobile`, 1,166 `person-noseatbelt`, and 1,639
`person-seatbelt` boxes; 84 selected representatives contain at least one source bbox that must
be clipped for display. None is automatically admitted. Run `training.prepare_v2_data_gaps` to
produce one review representative per inferred group, explicit phone-positive/negative queues,
the 500-item uncertainty UI queue, and a machine-readable action plan.
Reviewed ADT representatives are reserved as a source-disjoint external-test candidate. They
must not enter training or threshold tuning before the model is locked.

## Event semantics versus detector labels

- `phone` is a tight physical-phone detector box for an occupant inside a vehicle. A passenger phone can be labelled as a phone, but the event layer suppresses it because only the driver role violates the phone rule.
- `seatbelt_fastened` and `seatbelt_unfastened` are comparable upper-body boxes. Unfastened requires positive visible evidence; invisibility, darkness, crop, glare, or occlusion is `UNCERTAIN`.
- A person outside a vehicle is never a violation. Full traffic video must first produce a tracked vehicle/cabin ROI. The current safe demo accepts a declared vehicle/cabin crop and refuses unprepared raw-scene analysis.

## Download commands

Install provider clients locally, then explicitly accept the registry license:

```powershell
py -m pip install -r requirements-data.txt
py -m training.download_source --source-id kaggle_habbas_dms_v1 --accept-license Apache-2.0
py -m training.download_source --source-id mendeley_driver_risk_behavior_v1 --accept-license CC-BY-4.0
py -m training.download_source --source-id hf_anywaylabs_synthetic_dms --accept-license CC-BY-4.0
```

Kaggle authentication stays in Kaggle's local credential mechanism. Roboflow uses `ROBOFLOW_API_KEY` from the environment. Raw downloads are written once under `datasets/raw/<source>/<version>/` with per-file SHA256 and a tree hash; a second write to the same source/version is refused.
