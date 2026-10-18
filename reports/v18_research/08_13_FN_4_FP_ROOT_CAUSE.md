# 08. Detailed Root-Cause Analysis (13 FN, 4 FP)
### 13 False Negatives:
- **11/13 cases:** Full-frame downsampling (1280x720 -> 640x640) shrunk phone bounding box below detector receptive field. Auxiliary detector at original crop successfully detected 12/13 cases.
- **2/13 cases:** Phone box proposed, but action gate probability was under strict 0.80 cutoff.
- **Sub-second interaction:** Short events (<1.2s) dropped by default temporal duration requirements.
- **Label mismatch & engine reset:** Context `'HANDHELD'` versus `'HANDHELD_USE'` caused candidate to reset `TemporalEventEngine`, destroying accumulated observations.

### 4 False Positives:
- **2 cases:** Mounted phone on vent detected while driver adjusted AC.
- **1 case:** Passenger phone reaching into driver crop.
- **1 case:** Driver holding sunglasses case near ear.
