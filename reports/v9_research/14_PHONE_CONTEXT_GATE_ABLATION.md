# V9 phone-context gate ablation

Scope: **FN-selected development diagnostic only**. The table reports candidate yield, not detector
precision/recall or event metrics.

| Confidence | Raw proposals | Gated proposals | Removed | Raw case yield | Gated case yield | Human selections retained | Hard raw | Hard gated |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 0.02 | 226 | 127 | 99 | 13/13 | 12/13 | 12/12 | True | False |
| 0.10 | 71 | 60 | 11 | 12/13 | 11/13 | 11/12 | True | False |
| 0.20 | 43 | 41 | 2 | 11/13 | 11/13 | 9/12 | False | False |
| 0.25 | 32 | 30 | 2 | 11/13 | 10/13 | 8/12 | False | False |
| 0.35 | 21 | 21 | 0 | 10/13 | 10/13 | 7/12 | False | False |
| 0.50 | 16 | 16 | 0 | 8/13 | 8/13 | 6/12 | False | False |

At the 0.02 mining floor, the gate removes 99/226
proposals (43.81%). All removed proposals are `MOUNTED_OR_STATIC`;
all 12/12 human-selected
boxes are retained as `HANDHELD_USE`.

The adjudicated hard example `gB-9-phone-06` contains
12 proposals, all mounted/static. The raw 0.02 gate would
offer it as a candidate, while the context gate rejects every proposal. This directly addresses the
mounted cabin-camera failure without relabeling the visible phone as a negative.

The fusion API now accepts per-view context and exposes raw confidence, admitted confidence, and rejection
reasons. Strict mode fails closed for mounted, visible-but-not-used, possible, or unknown context. Legacy
mode remains unchanged until a caller supplies audited context and explicitly enables strict gating.

This positive-only FN sample has no true-negative denominator and no verified occupant tracks. It cannot
establish detector accuracy, event P/R/F1, false alarms/min, target attainment, or independent
generalization. No model was trained, and protected holdouts were not accessed.
