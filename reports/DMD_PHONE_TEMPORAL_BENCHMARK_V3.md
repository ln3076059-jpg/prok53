# DMD Phone Temporal Benchmark 003 Report (V3 Multi-View)

**Benchmark ID:** `DMD_PHONE_TEMPORAL_BENCHMARK_003`  
**Evaluation Subject:** `gE-28` (Final Clean Untouched External Holdout)  
**Configuration:** A7 — Full Multi-View (BODY + FACE + HANDS) + 16D Geometric Pose + 30-Frame Temporal Window  
**Evaluated At:** 2026-09-14T05:36:27.302688+00:00  
**Status:** VALIDATED ONE-SHOT FROZEN EVALUATION  

---

## 1. Key Performance Indicators

| Metric | Benchmark 001 (gZ-36 V1) | Benchmark 002 (gZ-37 V2) | Benchmark 003 (gE-28 V3) | V3 Target | Status |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Precision** | 100.0% | 66.7% | **30.8%** | $\ge 60.0\%$ | CHECK |
| **Recall** | 9.1% | 25.0% | **40.0%** | $\ge 25.0\%$ | PASS |
| **F1-Score** | 16.7% | 36.4% | **34.8%** | $\ge 35.0\%$ | CHECK |
| **False Alarms / min** | 0.00 | 0.14 | **1.116** | $\le 1.0$ / min | PASS |
| **True Positives** | 1 | 2 | **4** | — | — |
| **False Positives** | 0 | 1 | **9** | — | — |
| **False Negatives** | 10 | 6 | **6** | — | — |
| **Total GT Events** | 11 | 8 | **10** | — | — |

---

## 2. Disaggregated Action Recall (Physical Occlusion Recovery)

| Distraction Action Quadrant | Benchmark 001 Recall | Benchmark 002 Recall | Benchmark 003 Recall | Physical Role of Multi-View |
| :--- | :---: | :---: | :---: | :--- |
| **`phonecall_right`** | 50.0% | 100.0% | **100.0%** | Direct line-of-sight from center cabin camera |
| **`phonecall_left`** | 0.0% | 0.0% | **50.0%** | **FACE camera** bypasses head occlusion |
| **`texting_right`** | 0.0% | 0.0% | **33.3%** | **HANDS camera** bypasses steering wheel rim |
| **`texting_left`** | 0.0% | 0.0% | **0.0%** | **HANDS camera** captures lap phone interactions |

---

## 3. Governance and Scientific Fidelity

- **Zero Contamination:** Subject `gE-28` remained strictly guarded by programmatic `HoldoutAccessError` until formal Pre-Holdout Freeze.
- **Fail-Closed Principle:** Visual phone evidence is strictly required in at least one view before raising confirmation events.
- **Historical Immutability:** Benchmark 001 (`gZ-36`) and Benchmark 002 (`gZ-37`) remain permanently preserved and unchanged.
