# Roadwatch V4 Candidate Model Card

## Identity and status

- Model version: `ROADWATCH_V4_CANDIDATE_001`
- Config: `models/model_config_v4_candidate.yaml`
- Status: development-only, not V4 calibrated, not production approved
- Architecture: inherited YOLO phone/seatbelt detectors, YOLO seatbelt classifier, pose estimator,
  explicit context reasoning, fusion and temporal event logic

## Inherited component provenance

The candidate does not retrain weights. It reuses the locked V2 component artifacts whose SHA-256
values were verified against the lock during this upgrade. Dataset split hashes remain those in the
V2 lock. Human verification and production readiness are both false in that lock.

## Historical classification metrics

Seatbelt validation (`n=641`): fastened P/R/F1 0.8310/0.8571/0.8439; unfastened
0.7870/0.8195/0.8029; uncertain/occluded 0.3333/0.0400/0.0714. Macro-F1 is 0.5727 and balanced
accuracy (macro recall) is 0.5722. Historical confusion-matrix rows contain 361, 277 and 3 samples,
which conflicts with the JSON supports 350, 266 and 25; this provenance inconsistency must be
resolved before retraining or publishing a new seatbelt claim.

Abstention rate and false violation rate are not available in the checked-in calibration artifact.

## Intended use and limitations

The candidate is for controlled development evaluation of driver phone-use and seatbelt event
reasoning. It is not validated on a new independent holdout, has no fleet-domain calibration, and is
not suitable for autonomous enforcement. Uncertain context must remain abstained or human-reviewed.
