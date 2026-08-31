# Roboflow seatbelttraining v4 quality report

Status: **REVIEW_REQUIRED**

- Raw manifest: 16,745 files; tree SHA-256 `f93a14d1eb5f819685ef873dfd3102b70942cc82e1f6628271a9966a86095f79`.
- Images: 8,371; source boxes: 8,451; corrupt/invalid-label images: 0.
- Source classes: 1,805 `no-seatbelt`, 6,646 `seatbelt`.
- Canonical review proposals: 1,805 `seatbelt_unfastened`, 6,646 `seatbelt_fastened`.
- Exact duplicate images skipped during ingest: 0.
- Inferred source groups: 2,750; largest group: 183 images.
- The source's published train/valid/test layout leaks 147 inferred source groups across splits and is rejected.
- Human-approved images: 0.

Visual inspection confirmed that sampled boxes cover the driver/person rather than only a thin belt. This justifies review proposals, not automatic ground truth. Every box must still be checked for visible state evidence, comparable upper-body geometry, vehicle context, occupant role, augmentation lineage, and domain suitability. Final splitting must occur only after review and near-duplicate clustering.
