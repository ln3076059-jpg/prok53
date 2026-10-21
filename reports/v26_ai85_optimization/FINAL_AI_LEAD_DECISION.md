# ROADWATCH V26 — Final AI lead decision

Generated: 2026-10-21T06:39:18.496177+07:00

`DEVELOPMENT_TARGET_NOT_MET`. `INDEPENDENT_TARGET_NOT_ESTABLISHED`.
`production_approved=false`.

The cached reporter's missing-expiry defect is fixed; full backend parity is unproven.
Results use the unchanged
20-event development cohort, original annotations and frozen matcher:

| Run | V25 TP/FP/FN | V25 F1 | V26 TP/FP/FN | V26 Precision | V26 Recall | V26 F1 |
|---|---:|---:|---:|---:|---:|---:|
| E2 | 5/0/15 | 0.400000 | 12/13/8 | 0.480000 | 0.600000 | 0.533333 |
| E3 | 6/1/14 | 0.444444 | 12/15/8 | 0.444444 | 0.600000 | 0.510638 |
| E4 | 5/1/15 | 0.384615 | 12/14/8 | 0.461538 | 0.600000 | 0.521739 |
| E5 | 6/0/14 | 0.461538 | 12/14/8 | 0.461538 | 0.600000 | 0.521739 |

The best corrected run is E2 with TP=12, FP=13,
FN=8, precision=0.480000,
recall=0.600000 and F1=0.533333. The fix recovers
six E5 true positives and doubles E5 recall, but no V26 event metric exceeds 0.85.

The target cannot be established for detector and association components because exhaustive
pre-filter bbox labels and independent role ground truth are absent. Fitting, calibration,
fusion and boundary search remain blocked: fusion: explicit TRAINING_ALLOWED=true is required.
No holdout was accessed.

Tests: 94 passed, 0 failed, 0 errors,
0 skipped. Full raw-video D2 status is PARTIAL with
TP=4, FP=4, FN=7,
precision=0.500000, recall=0.363636, and
F1=0.421053. These are development diagnostics, not independent holdout evidence.
Completed raw subjects: gZ-36.
Cached results use D5 proposals, raw results use D2: this is not a controlled comparison.
Raw metrics precede the latest boundary collector; no improvement from that change is claimed.

Detailed diagnosis and primary-source comparisons:
[ROOT_CAUSE_AND_LITERATURE_REVIEW.md](ROOT_CAUSE_AND_LITERATURE_REVIEW.md).

Paper/GitHub comparison matrix and prioritized system gaps:
[RESEARCH_GITHUB_GAP_ANALYSIS.md](RESEARCH_GITHUB_GAP_ANALYSIS.md).
The machine-readable source register is
[research_github_matrix.json](research_github_matrix.json).
