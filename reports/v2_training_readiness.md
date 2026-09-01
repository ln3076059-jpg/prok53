# V2 training readiness

- Proposal detector training ready: `True`
- Governed training ready: `False`
- Production activation ready: `False`

## Blocking work

### PHONE_SEMANTIC_REVIEW_PENDING

9728 samples are still PENDING.

Required action: Apply append-only human review decisions; never promote proposals automatically.

### PHONE_SOURCE_GROUP_REVIEW_REQUIRED

4068 samples still use GROUP_REVIEW_REQUIRED source groups.

Required action: Resolve video/vehicle/person grouping before governed split certification.

### PHONE_SINGLE_SOURCE_DOMAIN

All samples come from one provider/domain.

Required action: Add a licensed, independently reviewed source with distinct cameras and vehicles.

### PHONE_NO_EXTERNAL_TEST_DOMAIN

The test split is not source-disjoint from training.

Required action: Create a frozen external test set from an unseen provider/camera domain.

### PHONE_DIVERSITY_METADATA_INCOMPLETE

Required diversity fields are incomplete: video_id, vehicle_id, person_id, camera_id.

Required action: Record real video, vehicle, person, and camera identifiers for every sample.

### PHONE_CONDITION_COVERAGE_UNPROVEN

Required adverse-condition coverage is not declared in the manifest.

Required action: Capture and review every condition defined by datasets/v2_capture_policy.yaml.

### SEATBELT_SEMANTIC_REVIEW_PENDING

4868 samples are still PENDING.

Required action: Apply append-only human review decisions; never promote proposals automatically.

### SEATBELT_SOURCE_GROUP_REVIEW_REQUIRED

4227 samples still use GROUP_REVIEW_REQUIRED source groups.

Required action: Resolve video/vehicle/person grouping before governed split certification.

### SEATBELT_SINGLE_SOURCE_DOMAIN

All samples come from one provider/domain.

Required action: Add a licensed, independently reviewed source with distinct cameras and vehicles.

### SEATBELT_NO_EXTERNAL_TEST_DOMAIN

The test split is not source-disjoint from training.

Required action: Create a frozen external test set from an unseen provider/camera domain.

### SEATBELT_DIVERSITY_METADATA_INCOMPLETE

Required diversity fields are incomplete: video_id, vehicle_id, person_id, camera_id.

Required action: Record real video, vehicle, person, and camera identifiers for every sample.

### SEATBELT_CONDITION_COVERAGE_UNPROVEN

Required adverse-condition coverage is not declared in the manifest.

Required action: Capture and review every condition defined by datasets/v2_capture_policy.yaml.

### PHONE_NEGATIVES_NOT_HUMAN_CONFIRMED

7513 samples contain zero phone boxes but remain proposals.

Required action: Review zero-box images so visible phones are not trained as background.

### SEATBELT_ROI_HARD_NEGATIVES_MISSING

Every seatbelt-detector image contains an upper-body box.

Required action: Add reviewed empty-cabin, seat, reflection, and non-occupant hard negatives.

### SEATBELT_CLASSIFIER_NOT_READY

The three-state classifier is missing a reviewed class in one or more splits.

Required action: Review independent UNCERTAIN_OR_OCCLUDED crops in train, val, and test.

### PRODUCTION_CALIBRATION_INCOMPLETE

Model configuration still contains markers: UNTRAINED, PLACEHOLDER, PENDING.

Required action: Calibrate thresholds/camera ROIs, lock weights, then run frozen test once.

## Warnings

### SEATBELT_CLASSIFIER_TRAIN_IMBALANCE

Fastened/unfastened train ratio is 2.00:1.

Recommended action: Use reviewed class balancing and report per-class recall; do not duplicate test data.

### EXTERNAL_RESUME_BACKUP_REQUIRES_CONFIGURATION

Automatic resume protects the working disk unless a second backup directory is supplied.

Recommended action: Pass -BackupDirectory on the portable launcher and place it on another disk or sync target.
