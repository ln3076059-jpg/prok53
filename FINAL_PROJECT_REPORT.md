# Final Project Report

Status: **ENGINEERING BASELINE COMPLETE; EXTERNAL SCIENTIFIC GATES OPEN**

## A. Data

Candidate and rejected sources are recorded with displayed licenses in `datasets/sources.yaml`. Downloaded candidate images: 25,851 (DMS 9,884; Sintes 8; Mendeley Driver Risk 1,232; AnywayLabs synthetic DMS 1,356; c3rl 5,000; Roboflow seatbelttraining v4 8,371). Every raw source has an immutable SHA-256 manifest. Human-approved images: 0. DMS contributes 2,548 unreviewed physical-phone boxes; Mendeley contributes 334 cellphone-behavior regions; AnywayLabs contributes 659 calling/texting behavior regions. Roboflow contributes 6,646 fastened and 1,805 unfastened upper-body/person proposals, but its published split leaks 147 inferred groups and is discarded. The c3rl 2,500/2,500 folder hints were visually downgraded to synthetic smoke-test/ablation only after samples from both classes showed geometric scenes rather than real occupants. Behavior regions, source state boxes, and folder hints are not governed ground truth before human review.

## B. Split

Canonical human-approved three-class train/validation/test: not created. A three-class proposal-only `mc_bootstrap_v1` now exists with 14,581/1,782/1,736 images. It contains 2,548 phone, 6,646 fastened-seatbelt, and 1,805 unfastened-seatbelt instances in total. Audit passes after grouping 118,786 pHash-near pairs into 5,055 leakage-safe components: zero SHA, source/base group, component, or near-duplicate cross-split overlap. A separate phone-only bootstrap remains available. These auxiliary splits must not be reported as final ground truth. Subject isolation remains `NOT_PROVABLE` until trusted metadata exists.

## C. Model

Planned final experiment: transfer learning from `yolo11s.pt`, MC_001, image size 960 with multi-scale training, seed 42, AMP, mosaic closed near the end, and validation-selected thresholds. A fail-closed `MC_BOOTSTRAP_001` three-class Kaggle bundle is ready and has passed an extracted-bundle preflight over 36,206 files; it is configured for 150 epochs, saves every epoch, produces an integrity-checked resume archive after every completed epoch, and packages an isolated-test evaluation. Its weights remain proposal/demo-only until human review. Training epochs completed, Kaggle GPU, best epoch, metrics, and model SHA remain unavailable until the web run.

## D. Validation

Phone, fastened, and unfastened P/R/F1/AP50/AP50-95: `NOT_RUN`. Macro F1, mAP50, and mAP50-95: `NOT_RUN`.

## E. Frozen test

`NOT_RUN`. The evaluator refuses test execution before a model lock exists.

## F. Performance

Model load time, p50/p95 latency, FPS, and memory: `NOT_RUN`. No real-time claim is made.

## G. System

Implemented: FastAPI endpoints, central model service, MySQL-compatible schema, safe uploads, ByteTrack integration, mandatory vehicle/cabin context, occupant-role association, temporal rules, event cooldown, evidence persistence, human review, CSV export, React dashboard, and annotation reviewer. The post-Kaggle phone-bootstrap handoff now verifies the recovery archive and weight SHA-256, installs an immutable proposal-only checkpoint, generates Mendeley proposals, and builds a focused review queue. Reviewer approval requires vehicle context plus resolved sample- and box-level occupant roles. End-to-end inference awaits locked weights. Phone events are driver-only; visible unfastened-seatbelt events cover every configured occupant role.

## H. Event evaluation

PHONE metrics: `NOT_RUN`. NO_SEATBELT metrics: `NOT_RUN`. Independent human event ground truth is required.

## I. Limitations

The primary blockers are human physical-phone reboxing, seatbelt upper-body semantic review, trustworthy subject/vehicle metadata, the Kaggle GPU runs, threshold calibration, frozen-test execution, and independent event ground truth. Generic COCO YOLO found phones in only 3 of 334 Mendeley cellphone-use frames at low confidence, confirming that domain-specific pretraining is necessary. Raw traffic scenes still require an upstream vehicle detector/tracker that yields cabin ROIs; the current safe runtime accepts vehicle/cabin crops and fails closed otherwise. Ambiguous belt visibility, mounted phones, passenger/driver association, cabin-camera geometry, low light, and cross-domain generalization remain material risks.
