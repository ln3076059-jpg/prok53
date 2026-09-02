# V2 Detailed Attention Triage Audit

- **ADMIN_CONFIRM_READY**: 0
- **AUTO_EXCLUDE**: 3900
- **CORRECTION_MANUAL**: 0
- **TOTAL**: 3900

## Reason Distribution
- GROUP_REVIEW_REQUIRED_SOURCE: 986
- OUT_OF_VEHICLE_OR_INVALID_OCCUPANT_CONTEXT: 2914

## Source Distribution
- datasets\manifests\v2_phone_positive_review.json: 986
- datasets\manifests\v2_phone_negative_review.json: 2777
- datasets\manifests\v2_phone_negative_priority_review.json: 137

## Spot Check (Seed 42)
- 100 samples checked, all PASS mapping to AUTO_EXCLUDE.
