MULTIMODEL V2 — model-assisted pending-approval exploratory training bundle
Governance: MODEL_ASSISTED_PENDING_APPROVAL; this is not HUMAN_APPROVED or production-ready.
Trainable components in this build: phone_detector, seatbelt_detector, seatbelt_classifier.
The classifier is included because its uncertainty evidence minimum is met.
1) Preflight: .\START_V2_PRETRAIN_REVIEW_TRAINING.ps1 -Mode preflight
2) Train: .\START_V2_PRETRAIN_REVIEW_TRAINING.ps1
The train command runs a one-epoch smoke suite first.
Resume: repeat the same command and working directory.
Second-disk backup: add -BackupDirectory X:\v2-pretrain-backup
Install once if needed: add -InstallDependencies
Component isolation: add -Only phone_detector, seatbelt_detector, or seatbelt_classifier.
No Internet is required for bundled YOLO base weights.
Training uses validation only; keep test frozen until model lock.
