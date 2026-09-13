# DMD Phone Detector Confidence Sweep Analysis

**Analysis Timestamp:** 2026-09-13T12:39:40+07:00  
**Target Data:** Development Subjects Pool (`gC-14` and `gZ-36`)  
**Total Evaluated Sequence Time:** 15.17 minutes (910 sampled frames at 1 FPS)  
**GT Positive Frames:** 512 frames | **Non-Target Control Frames:** 398 frames

---

## 1. Candidate Threshold Performance Sweep

| Confidence Threshold | GT Frame Hits | GT Frame Recall | Non-Target False Hits | False Alarms / Minute | Empirical Assessment |
| :---: | :---: | :---: | :---: | :---: | :--- |
| **0.05** | 129 | 25.20% | 15 | 0.99 / min | Maximum raw detector capture; boundary of allowable FA rate |
| **0.10** | 94 | 18.36% | 9 | 0.59 / min | High sensitivity; captures low-contrast hand-held phones |
| **0.15** | 69 | 13.48% | 3 | 0.20 / min | **Optimal operating point**: +72.5% recall over baseline with minimal noise |
| **0.20** | 50 | 9.77% | 2 | 0.13 / min | Strong signal-to-noise ratio; moderate recall loss |
| **0.25 (Baseline 001)**| 40 | 7.81% | 1 | 0.07 / min | Historical Benchmark 001 setting; misses partially occluded devices |
| **0.30** | 29 | 5.66% | 0 | 0.00 / min | Extreme precision, zero false alarms, but severe missed detections |
| **0.40** | 11 | 2.15% | 0 | 0.00 / min | Misses 97.8% of phone interaction frames |
| **0.50** | 4 | 0.78% | 0 | 0.00 / min | Captures only perfectly isolated, unoccluded front-facing screens |

---

## 2. Threshold Selection Recommendation for Temporal Pipeline V2

1. **Candidate Threshold ($0.15$):** Setting the detector candidate threshold to $0.15$ increases positive frame detection from 40 to 69 frames (+72.5%), while keeping non-target false hits to only 3 frames across 15+ minutes ($0.20$ false alarms/minute).
2. **Context Filtering Synergies:** The 3 non-target detections at threshold $0.15$ are filtered out by the driver cabin ROI and hand/face proximity gate, preventing them from generating spurious temporal events.
3. **Temporal Integration:** Lowering candidate threshold to $0.15$ combined with multi-frame aggregation allows the temporal engine to accumulate sufficient observations during partially occluded interactions.
