# Roadwatch V6 Final Completion Report

## Outcome

V6 completed a code-backed investigation of class collapse, four controlled LOSO ablations, fixed-threshold event evaluation, case-level false-positive audit, independent seatbelt metric recomputation, runtime/config hardening, software verification and a real local performance benchmark. Selected V6 result: không có V6 candidate vượt toàn bộ gate đăng ký trước.

The central training finding is that V5 applied both replacement balancing and weighted loss even though SAFE was not scarce. V6 separates those mechanisms and evaluates a hierarchical alternative. The central validity finding is that V5 threshold calibration was training-subject in-sample; V6 therefore freezes the threshold instead of optimizing against outer labels.

Seatbelt is now measured rather than NOT_RUN: raw accuracy 80.97%, macro-F1 57.28%, balanced accuracy 57.22%, ECE 0.0846. Its uncertain class remains inadequate and blocks any strong claim.

Software verification recorded 5 passing and 2 failing checks. Full backend cabin processing measured 1.833 FPS on this machine. See Sections 10–11 for exact failures and performance interpretation.

An additional expanded backend/pipeline suite recorded 112 passed and 1 skipped tests with no failures or errors.

## Thesis chapter mapping

- Chapter 1: scope, objectives and non-roadside limitation — Sections 01, 13.
- Chapter 2: related work and inherited references — `references.bib` plus V5 literature review.
- Chapter 3: repository architecture, data and workflow — Sections 02, 03.
- Chapter 4: reproducible methods, ablations, seatbelt, tests and performance — Sections 04–12.
- Chapter 5: conclusion and future work — this report and Section 13.

## Final readiness

**Share with caveats within the reviewed thesis-development scope.** The software is demonstrable and the measurements are traceable, but the project is not production-approved and does not establish roadside-camera accuracy. No V6 model was activated, no consumed holdout was tuned, and no push or merge to `main` was performed.
