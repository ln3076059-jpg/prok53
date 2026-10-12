# Roadwatch V4 AI Failure Analysis

## Evidence boundary

This report uses checked-in historical artifacts only. Development videos/features are absent from
the current working tree, so case-level inference was not rerun and no new root cause was asserted.
Consumed holdouts `gZ-37` and `gE-28` were not accessed for tuning.

## Verified historical symptoms

- V3 holdout `gE-28`: TP 4, FP 9, FN 6; precision 30.8%, recall 40.0%, F1 34.8%,
  1.116 false alarms/min (`DMD_PHONE_TEMPORAL_BENCHMARK_V3.json`).
- V3.1 B5 development pool: TP 3, FP 5, FN 26; precision 37.5%, recall 10.3%, F1 16.2%,
  0.227 false alarms/min (`v31_ablation_results.json`). These values are not comparable to the
  V3 holdout because the evaluation sets differ.
- Seatbelt `uncertain_or_occluded` historical validation recall is 4.0% with support 25.

## Root-cause taxonomy

The checked-in V3.1 analysis attributes 15 development false positives as follows:

| Category | Historical count | V4 response |
|---|---:|---|
| Duplicate event split | 5 | Stable occupant identity across phone track-ID changes; explicit temporal state |
| Ungated multi-view auxiliary evidence | 4 | Six-state phone context and strict abstention policy |
| Detector false object | 3 | Require physical phone plus use context; no retraining claim |
| Pose/action confusion | 2 | Hand/face proximity cannot create a phone when no physical detection exists |
| Stale track/ghost | 1 | Candidate config reduces occlusion bridge from 4 s to 2 s |

Other required taxonomy classes—wrong person, steering-wheel occlusion, lap use, mounted phone,
watch/sunglasses, multi-view disagreement, early termination, onset delay and seatbelt occlusion—are
represented in policy/tests where possible, but cannot be assigned new incident counts without the
missing frame-level evidence.

## V4 context decision contract

Runtime reasoning now distinguishes `NO_PHONE`, `PHONE_VISIBLE_NOT_USED`, `MOUNTED_PHONE`,
`HANDHELD_PHONE_USE`, `POSSIBLE_PHONE_USE`, and `UNKNOWN_CONTEXT`. Only handheld evidence may
remain `PENDING`; possible use is `NEEDS_REVIEW`; visible-not-used and mounted states produce no
event; unknown context abstains in strict V4 mode.

## Missing evidence fields

The requested inspectable record needs video/subject ID, timestamp, truth, prediction, confidence,
pose geometry, per-view confidence, fusion score, temporal state and track ID. Historical aggregate
reports do not contain this complete row-level set. The V4 experiment preflight therefore blocks
rather than synthesizing it.
