# V10 model checkpoint inventory

Inventory contains **64** checkpoint files totaling
0.40 GiB. Checkpoints were hashed as bytes and were not deserialized.

| Runtime role | Path | Bytes | SHA-256 | Provenance |
| --- | --- | --- | --- | --- |
| active_phone_detector | models/active/v2/phone_detector.pt | 19209626 | 840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54 | PROVENANCE_INCOMPLETE |
| active_seatbelt_detector | models/active/v2/seatbelt_detector.pt | 19345306 | 361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500 | PROVENANCE_INCOMPLETE |
| active_seatbelt_classifier | models/active/v2/seatbelt_classifier.pt | 11028162 | e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4 | PROVENANCE_INCOMPLETE |
| active_combined | models/active/best.pt | 19209626 | 840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54 | PROVENANCE_INCOMPLETE |
| auxiliary_coco | models/auxiliary/yolo11n.pt | 5613764 | 0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1 | PUBLIC_PRETRAINED_LINEAGE_ASSERTED_LOCALLY_LICENSE_NOT_REVERIFIED |
| auxiliary_pose | models/auxiliary/yolo11n-pose.pt | 6255593 | 869e83fcdffdc7371fa4e34cd8e51c838cc729571d1635e5141e3075e9319dc0 | PUBLIC_PRETRAINED_LINEAGE_ASSERTED_LOCALLY_LICENSE_NOT_REVERIFIED |

Provenance-state counts: `{'LOCAL_METADATA_PRESENT_NOT_RIGHTS_APPROVAL': 47, 'PROVENANCE_INCOMPLETE': 14, 'PUBLIC_PRETRAINED_LINEAGE_ASSERTED_LOCALLY_LICENSE_NOT_REVERIFIED': 3}`.

`LOCAL_METADATA_PRESENT_NOT_RIGHTS_APPROVAL` means an args/report/metadata/lock file exists locally; it
does not prove dataset rights. Files without sufficient sidecars remain `PROVENANCE_INCOMPLETE`. Architecture
and class mapping are taken only from explicit configs/metadata, never guessed from a filename such as
`best.pt`. No checkpoint was downloaded, loaded, promoted, or activated.
