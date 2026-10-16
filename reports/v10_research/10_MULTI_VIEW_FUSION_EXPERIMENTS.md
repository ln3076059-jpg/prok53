# V10 multi-view fusion experiments

The only valid frozen comparison changes phone context while holding observations, temporal configuration,
ground truth, and matcher fixed. On two repeatedly used development subjects, context-naive replay is
TP=6, FP=4, FN=14, F1=40.00%,
FA/min=0.2620. Context-aware replay is TP=6,
FP=3, FN=14, F1=41.38%,
FA/min=0.1965.

Measured delta: recovered TP=0, lost TP=0, removed FP=1, new FP=0, FN change=0. The historical artifact did
not persist event IDs for that removed FP, so ID-level traceability is **NOT AVAILABLE** and this result is
ineligible for promotion. `gC-14` has BODY only; missing FACE/HANDS is represented as missing information,
never negative evidence. G1/G2/G4 and a fully validated multi-view candidate remain unmeasured.
