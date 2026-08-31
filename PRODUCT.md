# Product

<!-- impeccable:product-schema 1 -->

## Platform

web

## Stack

Confirmed by the project brief: React with TypeScript, FastAPI, MySQL, Ultralytics YOLO11s, OpenCV, and a single canonical Kaggle GPU training workflow.

## Users

- Safety reviewers inspect evidence and confirm or reject detected violations.
- Operations analysts upload recordings, monitor analysis jobs, search events, export records, and review system statistics.
- ML engineers govern datasets, run the single approved training experiment, lock artifacts, and audit reproducibility.

## Product Purpose

Detect visible phones and explicit fastened or unfastened seatbelt states with one three-class detector, then apply vehicle context, occupant-role association, and temporal confirmation to create reviewable PHONE and NO_SEATBELT events.

Success requires legal data provenance, group-clean evaluation, reproducible model artifacts, usable evidence, and an end-to-end application. Detector metrics and event metrics remain separate.

## Positioning

One governed YOLO11s detector produces object and upper-body state observations on a vehicle/cabin crop. A transparent event layer associates detections with configured occupant regions, confirms them over time, preserves evidence, and keeps uncertain cases in human review.

## Operating Context

Inputs are uploaded images or video files and, where safely configured, camera streams. Reviewers work from event queues, evidence frames, source metadata, model versions, and review history. ML engineers work from immutable raw assets, source manifests, review decisions, group-aware splits, frozen-test controls, and a locked experiment record.

## Capabilities and Constraints

- Canonical detection classes are `phone`, `seatbelt_fastened`, and `seatbelt_unfastened`.
- Phone boxes cover the visible physical phone. Seatbelt-state boxes cover comparable person upper-body regions.
- Missing or unclear belt evidence is never converted to `seatbelt_unfastened`.
- A detection without a tracked vehicle/cabin context never creates an event.
- Driver phone use is a violation; passenger phone use is not. Visible unfastened occupants, including passengers, are violations.
- Only human-approved seatbelt annotations can enter the governed dataset.
- The primary experiment is MC_001 using `yolo11s.pt`, image size 640, seed 42, and one serious Kaggle GPU run.
- Validation alone selects thresholds. The frozen test is used once after model lock.
- Runtime violations are `PHONE` and `NO_SEATBELT`; review states are `PENDING`, `CONFIRMED`, `REJECTED`, and `NEEDS_REVIEW`.
- Real dataset acquisition, human semantic review, Kaggle execution, trained weights, and measured metrics are not yet available and must not be simulated.

## Evidence on Hand

The attached master specification is the sole confirmed project brief. No approved dataset, trained model, human event ground truth, production camera, customer claims, or measured performance evidence was supplied.

## Product Principles

- Prefer traceable evidence over confident guesses.
- Preserve ambiguity for human review.
- Keep train, validation, frozen test, and event evaluation boundaries explicit.
- Make every model and dataset claim reproducible by hash and version.
- Use one simple deployable model and one canonical training workflow.

## Accessibility & Inclusion

The web interfaces must support keyboard operation, visible focus, readable contrast, reduced motion, clear status text that does not rely on color alone, and responsive layouts suitable for review work.
