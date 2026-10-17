# V11 fusion ablation

The only available frozen event replay is inherited from V9 and covers two development subjects. Context-naive fusion records TP=6, FP=4, FN=14, F1=40.00%, FA/min=0.2620. Context-aware fusion records TP=6, FP=3, FN=14, F1=41.38%, FA/min=0.1965.

Measured aggregate delta is recovered TP=0, lost TP=0, removed FP=1, new FP=0, and FN change=0. The inherited artifact did not retain the removed FP event ID, so ID-level evidence is NOT AVAILABLE. G1-G6 were not rerun because the V11 clip candidate does not yet produce calibrated event evidence and real occupant association remains UNKNOWN.

Future fusion must persist raw object, action, geometry, association, fused confidence, rejection reasons, and source views per event. Missing views remain UNKNOWN. Passenger and mounted devices cannot create driver-phone alerts. V11 does not use a new hard gate or claim event improvement from clip metrics.
