# V14 event replay results

Real candidate replay is `BLOCKED_MISSING_FITTED_HEAD_AND_VERIFIED_ASSOCIATION`. The frozen matcher, event denominators and ground truth were unchanged. V14 therefore retains the 16/4/13 controlled comparator and does not manufacture events from detector yield or synthetic model weights.

The opt-in candidate still records stable event ID, source key, actual emission time, model/config hashes, raw phone evidence, uncalibrated action scores, association state and rejection reasons. A legitimate replay additionally requires a fitted E3 head/scaler and verified driver association. Recovered/lost/new/removed event IDs, fragmentation, short-event recall and real alert latency are not newly measured.
