# Final Project Report

Status: **V2 BASELINE EVIDENCE VERIFIED; EXTERNAL SCIENTIFIC GATES OPEN**

## A. Data

Candidate and rejected sources are recorded with displayed licenses in `datasets/sources.yaml`. Downloaded candidate images: 25,851 (DMS 9,884; Sintes 8; Mendeley Driver Risk 1,232; AnywayLabs synthetic DMS 1,356; c3rl 5,000; Roboflow seatbelttraining v4 8,371). Every raw source has an immutable SHA-256 manifest. Human-approved images: 0. DMS contributes 2,548 unreviewed physical-phone boxes; Mendeley contributes 334 cellphone-behavior regions; AnywayLabs contributes 659 calling/texting behavior regions. Roboflow contributes 6,646 fastened and 1,805 unfastened upper-body/person proposals, but its published split leaks 147 inferred groups and is discarded. The c3rl 2,500/2,500 folder hints were visually downgraded to synthetic smoke-test/ablation only after samples from both classes showed geometric scenes rather than real occupants. Behavior regions, source state boxes, and folder hints are not governed ground truth before human review.

The machine-readable V2 diversity audit measures 9,728 phone proposal samples (one provider,
3,616 declared source groups), 4,868 upper-body proposal samples (one provider, 2,750 groups), and
4,929 classifier crops. Camera, video, vehicle, and person identities are absent from all three
proposal manifests; subject-disjoint status is `NOT_PROVABLE`. Image dimensions were measured from
the available files, while semantic conditions were not inferred. The prioritized phone-negative
queue contains 3,013 `PENDING` rows (679 priority-1, 1,329 priority-2, 1,005 priority-4) and zero
human approvals. All ten seatbelt hard-negative capture scenarios remain absent.

An append-only Review 1 lane has now visually inspected the first 200 phone-negative candidates.
It records 63 `REVIEW1_ACCEPTED_PROPOSAL` Tier-B decisions and 137
`REVIEW1_REJECTED_PROPOSAL` Tier-C decisions, with source/image evidence hashes and admin
delegation provenance. This reduces the immediate hard-case/manual-attention queue to 2,950 but
does not convert any record to HUMAN approval. The derived bootstrap lane is explicitly
`MODEL_ASSISTED`, `ADMIN_DELEGATED`, `NOT_HUMAN_APPROVED`, and `NOT_GOVERNED`; the governed lane
contains zero records.

## B. Split

Canonical human-approved three-class train/validation/test: not created. A reduced three-class proposal-only `mc_bootstrap_v2_6500` now exists with 6,500/1,782/1,736 images. It retains every phone-positive and unfastened-positive train image, selects difficult/context-diverse fastened and negative frames, and preserves all 3,288 original train source groups. It contains 2,548 phone, 3,124 fastened-seatbelt, and 1,805 unfastened-seatbelt instances in total. Audit passes with 27,616 retained pHash-near pairs and zero SHA, source/base group, component, or near-duplicate cross-split overlap. The full `mc_bootstrap_v1` and separate phone-only bootstrap remain available. These auxiliary splits must not be reported as final ground truth. Subject isolation remains `NOT_PROVABLE` until trusted metadata exists.

## C. Model

Planned final experiment: transfer learning from `yolo11s.pt`, MC_001, image size 960 with multi-scale training, seed 42, AMP, mosaic closed near the end, and validation-selected thresholds. A fail-closed `MC_BOOTSTRAP_001` reduced-data Kaggle bundle is ready and has passed an extracted-bundle preflight over 20,045 files; it is configured for 150 epochs with stabilized AdamW learning rate, rejects non-finite labels/losses, saves every epoch, produces an integrity-checked resume archive after every completed epoch, and packages an isolated-test evaluation. Its weights remain proposal/demo-only until human review. Training epochs completed, Kaggle GPU, best epoch, metrics, and model SHA remain unavailable until the remote run.

## D. Validation

Canonical automated evaluation via CPU 30-core inference generated raw validation evidence (V2_baseline_001):
* **Phone Detector (Val 302 imgs):** Precision 97.0%, Recall 82.7%, mAP50 94.1%, mAP50-95 68.3%
* **Seatbelt Detector (Val 636 imgs):** Precision 92.6%, Recall 87.9%, mAP50 93.9%, mAP50-95 50.3%
* **Seatbelt Classifier (Val 641 imgs, 3-class):** Top-1 Accuracy 81.0%, Top-5 Accuracy 100.0%
Raw confusion matrices and PR curves successfully archived.

*Note: A previous non-canonical evaluation generated slightly higher raw metrics (Phone: 97% mAP50, Seatbelt: 93% mAP50), but dataset identities and classes lacked the required strict subsets.* 

## E. Frozen test

Canonical execution of the frozen test split produced the following baseline evidence:
* **Phone Detector (Test 302 imgs, 139 instances):** Precision 90.2%, Recall 86.2%, mAP50 90.5%, mAP50-95 65.5%
* **Seatbelt Detector (Test 614 imgs, 617 instances):** Precision 93.8%, Recall 88.2%, mAP50 92.7%, mAP50-95 47.7%
* **Seatbelt Classifier (Test 621 imgs, 3-class):** Top-1 Accuracy 78.9%, Top-5 Accuracy 100.0%
Integration blocker lifted; model status transitioned to VERIFIED.

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
scientific gate. GitHub Actions run
[`33591019254`](https://github.com/ln3076059-jpg/prok53/actions/runs/33591019254) passed for
the verified code commit `1638654`.

## J. Pre-training engineering freeze

The final code audit found and fixed ten engineering issue groups: single-process inference
serialization, failed-source cleanup/input validation, evidence-root containment, explicit and
idempotent human-review provenance, calibration-to-model/threshold binding, complete configured
component locking, post-evaluation hash verification, production secret safety, and CSV/MIME
hardening, plus a repository-root-safe reviewer launcher. Local verification passes 125/125
tests; these are tooling checks, not model metrics.

The repository is ready to begin controlled human review, but governed training and production
remain false. Durable multi-process queuing, production RTSP behavior and target-hardware
performance are not claimed. GitHub reports `main` is unprotected, so branch protection remains
`BLOCKED_BY_GITHUB_SETTINGS`. See `reports/PRE_TRAIN_ENGINEERING_FREEZE.md`.
