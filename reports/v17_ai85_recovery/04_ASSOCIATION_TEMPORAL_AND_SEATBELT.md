# Association, temporal events and seatbelt

`training/dmd/candidate_flow.py` provides ordered stage decisions with mandatory reason codes,
GT-positive rejection counts, admitted-negative counts and UNKNOWN coverage. Accuracy analysis
is INSTRUMENTATION_READY_ACCURACY_BLOCKED because no authorized cohort has occupant-role, object and event truth
together. NEEDS_REVIEW remains outside confirmed TP metrics.

The candidate temporal configuration requires at least
0.50s and
2 observations. Evidence spans below that
duration cannot activate. V17 did not tune the threshold or matcher because a valid replay set
does not exist.

Seatbelt remains blocked by 500 pending human reviews. No automatic approval or
training occurred. The inherited development Macro-F1 is
**58.52%** and UNCERTAIN recall is
**4.00%**.
