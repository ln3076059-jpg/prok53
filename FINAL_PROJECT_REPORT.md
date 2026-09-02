# Final Project Report

Status: **ENGINEERING BASELINE COMPLETE; EXTERNAL SCIENTIFIC GATES OPEN**

## A. Data

Candidate and rejected sources are recorded with displayed licenses in `datasets/sources.yaml`. Downloaded candidate images: 25,851 (DMS 9,884; Sintes 8; Mendeley Driver Risk 1,232; AnywayLabs synthetic DMS 1,356; c3rl 5,000; Roboflow seatbelttraining v4 8,371). Every raw source has an immutable SHA-256 manifest. Human-approved images: 0. DMS contributes 2,548 unreviewed physical-phone boxes; Mendeley contributes 334 cellphone-behavior regions; AnywayLabs contributes 659 calling/texting behavior regions. Roboflow contributes 6,646 fastened and 1,805 unfastened upper-body/person proposals, but its published split leaks 147 inferred groups and is discarded. The c3rl 2,500/2,500 folder hints were visually downgraded to synthetic smoke-test/ablation only after samples from both classes showed geometric scenes rather than real occupants. Behavior regions, source state boxes, and folder hints are not governed ground truth before human review.

The machine-readable V2 diversity audit measures 9,728 phone proposal samples (one provider,
3,616 declared source groups), 4,868 upper-body proposal samples (one provider, 2,750 groups), and
4,929 classifier crops. Camera, video, vehicle, and person identities are absent from all three
proposal manifests; subject-disjoint status is `NOT_PROVABLE`. Image dimensions were measured from
the available files, while semantic conditions were not inferred. The prioritized phone-negative
queue contains 3,013 `PENDING` rows (679 priority-1, 1,329 priority-2, 1,005 priority-4) and zero
human approvals. All ten seatbelt hard-negative capture scenarios remain absent.

## B. Split

Canonical human-approved three-class train/validation/test: not created. A reduced three-class proposal-only `mc_bootstrap_v2_6500` now exists with 6,500/1,782/1,736 images. It retains every phone-positive and unfastened-positive train image, selects difficult/context-diverse fastened and negative frames, and preserves all 3,288 original train source groups. It contains 2,548 phone, 3,124 fastened-seatbelt, and 1,805 unfastened-seatbelt instances in total. Audit passes with 27,616 retained pHash-near pairs and zero SHA, source/base group, component, or near-duplicate cross-split overlap. The full `mc_bootstrap_v1` and separate phone-only bootstrap remain available. These auxiliary splits must not be reported as final ground truth. Subject isolation remains `NOT_PROVABLE` until trusted metadata exists.

## C. Model

Planned final experiment: transfer learning from `yolo11s.pt`, MC_001, image size 960 with multi-scale training, seed 42, AMP, mosaic closed near the end, and validation-selected thresholds. A fail-closed `MC_BOOTSTRAP_001` reduced-data Kaggle bundle is ready and has passed an extracted-bundle preflight over 20,045 files; it is configured for 150 epochs with stabilized AdamW learning rate, rejects non-finite labels/losses, saves every epoch, produces an integrity-checked resume archive after every completed epoch, and packages an isolated-test evaluation. Its weights remain proposal/demo-only until human review. Training epochs completed, Kaggle GPU, best epoch, metrics, and model SHA remain unavailable until the remote run.

## D. Validation

Phone, fastened, and unfastened P/R/F1/AP50/AP50-95: `NOT_RUN`. Macro F1, mAP50, and mAP50-95: `NOT_RUN`.

## E. Frozen test

`NOT_RUN`. The evaluator refuses test execution before a model lock exists.

## F. Performance

Decode FPS; vehicle, cabin, behavior, pose, seatbelt-classifier and fusion latency; total p50/p95;
GPU VRAM; and CPU RAM: `NOT_RUN`. The benchmark tooling is implemented, but no target-hardware
video run has been supplied. No real-time claim is made.

## G. System

Implemented: FastAPI endpoints, central model service, MySQL-compatible schema, safe uploads, ByteTrack integration, mandatory vehicle/cabin context, occupant-role association, temporal rules, event cooldown, evidence persistence, human review, CSV export, React dashboard, and annotation reviewer. The post-Kaggle phone-bootstrap handoff now verifies the recovery archive and weight SHA-256, installs an immutable proposal-only checkpoint, generates Mendeley proposals, and builds a focused review queue. Reviewer approval requires vehicle context plus resolved sample- and box-level occupant roles. End-to-end inference awaits locked weights. Phone events are driver-only; visible unfastened-seatbelt events cover every configured occupant role.

## H. Event evaluation

PHONE metrics: `NOT_RUN`. NO_SEATBELT metrics: `NOT_RUN`. Independent human event ground truth is required.
Vehicle/cabin/occupant/phone association accuracy and pose availability: `NOT_RUN`. The evaluator
is implemented, but independently reviewed association annotations are required. Event truth now
has an explicit immutable freeze artifact bound to the frozen external-test SHA-256. Event
evaluation requires an ACTIVE governed model lock, records every input hash, and refuses to
overwrite the first result. These refusal gates do not create a metric; all event results remain
`NOT_RUN`.

## I. Limitations

The primary blockers are human physical-phone reboxing, seatbelt upper-body semantic review, trustworthy subject/vehicle metadata, the Kaggle GPU runs, threshold calibration, frozen-test execution, and independent event ground truth. Generic COCO YOLO found phones in only 3 of 334 Mendeley cellphone-use frames at low confidence, confirming that domain-specific pretraining is necessary. Raw traffic support now performs vehicle tracking followed by explicit cabin localization; it fails closed when the vehicle is untracked or the cabin is unknown. Ambiguous belt visibility, mounted phones, passenger/driver association, cabin-camera geometry, low light, and cross-domain generalization remain material risks.

CI now checks Ruff, Python compilation, tests, FastAPI import, and the frontend build without GPU
training or dataset/model downloads. CI correctness does not supply model accuracy or clear any
scientific gate.

## J. Pre-training engineering freeze

The final code audit found and fixed nine engineering issue groups: single-process inference
serialization, failed-source cleanup/input validation, evidence-root containment, explicit and
idempotent human-review provenance, calibration-to-model/threshold binding, complete configured
component locking, post-evaluation hash verification, production secret safety, and CSV/MIME
hardening. Local verification passes 124/124 tests; these are tooling checks, not model metrics.

The repository is ready to begin controlled human review, but governed training and production
remain false. Durable multi-process queuing, production RTSP behavior and target-hardware
performance are not claimed. GitHub reports `main` is unprotected, so branch protection remains
`BLOCKED_BY_GITHUB_SETTINGS`. See `reports/PRE_TRAIN_ENGINEERING_FREEZE.md`.
