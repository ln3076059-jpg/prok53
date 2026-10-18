# Data ingestion and QA

V16 reused `training/download_source.py`, `training/ingest_dataset.py` and
`training/split_dataset.py`. CLI defaults now point at the repository's actual `dataset/`
layout. The downloader now requires the exact `DOWNLOAD_AUTHORIZED` registry state, a verified
license and an explicit `download` operation before any network function is called.

| Existing manifest | Rows | Unique SHA-256 | Groups | Approved |
|---|---:|---:|---:|---:|
| `dataset/manifests/ingest_adt_seatbelt_v1.jsonl` | 2704 | 2704 | 2704 | 0 |
| `dataset/manifests/ingest_anywaylabs_synthetic_dms.jsonl` | 1356 | 1356 | 554 | 0 |
| `dataset/manifests/ingest_c3rl_seatbelt_v1.jsonl` | 5000 | 5000 | 5000 | 0 |
| `dataset/manifests/ingest_dms.jsonl` | 9884 | 9728 | 4230 | 0 |
| `dataset/manifests/ingest_mendeley_driver_risk_v1.jsonl` | 1232 | 1232 | 5 | 0 |
| `dataset/manifests/ingest_roboflow_seatbelttraining_v4.jsonl` | 8371 | 8371 | 2750 | 0 |
| `dataset/manifests/ingest_sintes.jsonl` | 8 | 8 | 8 | 0 |

The inventory contains **28555** records and
**0** approved training records. V16 added zero original
images, subjects, videos and events. Existing manifests retain exact SHA-256, perceptual hashes,
source-group IDs, decode checks, raw labels and pending review states. Raw media was not
committed, overwritten or copied into this report bundle.
