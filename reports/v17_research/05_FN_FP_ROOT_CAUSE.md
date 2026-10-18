# 05. Root-Cause Audit of 13 False Negatives & 4 False Positives

## 1. Audit Methodology
The 13 historical FN events and 4 FP events from the V15/V16 baseline were traced across the seven operational candidate stages:
`DETECTED` -> `DRIVER_ASSOCIATED` -> `HANDHELD_EVIDENCE` -> `ACTION_SUPPORTED` -> `FUSION_ADMITTED` -> `TEMPORAL_ACTIVATED` -> `EVENT_MATCHED`.

## 2. False Negative (FN=13) Breakdown
1. **Sub-second Duration Filter (5 cases, ~38.5%):**
   * *Mechanism:* Events with physical interaction lasting <1.2s were rejected by `TemporalEventEngine.min_positive_seconds = 1.2s`.
   * *Evidence:* Driver glances at phone and puts it down within 0.8s - 1.0s. At 2 FPS, only 1-2 frames capture the action. `len(observations) < min_observations (3)` or span < 1.2s triggers automatic drop.
2. **Detector Small-Object Proposal Miss (4 cases, ~30.8%):**
   * *Mechanism:* Phone placed on driver's lap or partially occluded by hand/steering wheel produced detector confidence below `confidence_floor = 0.20`.
   * *Evidence:* Full-frame 640x640 resize downscales a 24x30 px phone to ~8x10 px, obliterating convolutional feature maps.
3. **Driver Association Ambiguity (2 cases, ~15.4%):**
   * *Mechanism:* Phone held near center console triggered `AMBIGUOUS_OCCUPANT_CONTAINMENT` because bounding box overlap between driver ROI and passenger seat margin was within `ambiguity_margin = 0.15`.
4. **Cadence Gap & Stream Reset (2 cases, ~15.4%):**
   * *Mechanism:* Video frame drop caused timestamp difference to exceed `cadence_tolerance = 0.02s`, causing `OnlineE3` and `TemporalEventEngine` to perform an unrecoverable `STREAM_RESET`.

## 3. False Positive (FP=4) Breakdown
1. **Mounted Phone / GPS Confusion (2 cases, 50.0%):**
   * *Mechanism:* Phone mounted on windshield air vent was detected with high confidence (>0.70) while driver's hand passed near it to adjust AC.
2. **Passenger Phone Spillover (1 case, 25.0%):**
   * *Mechanism:* Passenger reaching across cabin caused passenger-associated phone to enter driver torso crop.
3. **Non-Phone Action Conflict (1 case, 25.0%):**
   * *Mechanism:* Driver scratching cheek/ear with a dark object (sunglasses case) produced false phone proposal paired with high calling pose score.
