# ROADWATCH V19 — Root-Cause Analysis of 13 FN and 4 FP Events
## 1. Why Event Recall Was 55.17% Despite 94.48% Detector mAP
1. **Downscaling Bottleneck (Stage 1):** Full-frame 1280x720 downscaled to 640x640 reduces handheld phone boxes to <20px, falling below feature extractor resolution (blocked 11/13 FN).
2. **Context Mismatch (Stage 2):** Pre-V18 string comparison `phone_context == "HANDHELD_USE"` rejected valid observations.
3. **Temporal Engine Reset (Stage 3):** Pre-V18 re-creation of `TemporalEventEngine` on momentary occlusions wiped accumulated evidence.
4. **Sub-second Event Windowing (Stage 4):** Events <1.2s were dropped by default temporal duration thresholds.

## 2. False Positive Mechanisms
1. **Mounted Phone Spillover:** Static phones mounted near dashboard triggered proximity alerts if hand passed near mount.
2. **Micro-Event Fragmentation:** Wiping engine state broke contiguous events into separate un-cooldown-merged alerts.
