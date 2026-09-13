# DMD Phone Temporal Benchmark (Held-Subject Evaluation)

**Status:** COMPLETE  
**Benchmark Classification:** `DMD_EXTERNAL_DEVELOPMENT_BENCHMARK`  
**Governance Boundary:** Development temporal benchmark on held DMD subject. This is NOT the canonical frozen model test, nor a production holdout.  
**Held Subjects Evaluated:** 36  
**Total Evaluated Duration:** 8.60 minutes (516.0 seconds)  
**Phone Model SHA256:** `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`  
**Temporal Config Hash:** `5d70dc287251f79cdf3ecf5019f007cf80d4a7a4162bb49e1bf22e496549199c`  

---

## 1. Temporal Event Performance Metrics

| Metric | Measured Value | Target / Baseline Standard |
|---|---|---|
| **PHONE Event Precision** | **100.0%** | $\ge 75.0\%$ |
| **PHONE Event Recall** | **9.1%** | $\ge 70.0\%$ |
| **PHONE Event F1 Score** | **16.7%** | $\ge 72.0\%$ |
| **False Alarms per Minute** | **0.00** | $\le 1.00$ / min |
| **Mean Event Start Latency** | **+10.38s** | $\le 2.50$s onset tolerance |
| **Mean Event End Latency** | **-40.46s** | $\le 3.50$s offset tolerance |
| **True Positive Events** | **1** | Total GT: 11 |
| **False Positive Events** | **0** | Unmatched predictions |
| **False Negative Events** | **10** | Missed GT intervals |

---

## 2. Association & Auxiliary Diagnostics

| Diagnostic Metric | Measured Rate | Evaluation Note |
|---|---|---|
| **Pose Availability Rate** | 100.0% | Upper-body keypoints detected |
| **Unknown Occupant Rate** | 0.0% | Cabin subject role resolution |
| **Phone Context Unknown Rate** | 0.0% | Phone without hand/face binding |
| **Independent Association Accuracy** | `NOT_EVALUABLE` | DMD lacks independent occupant assignment truth |

---

## 3. Event Matching Policy
- **Minimum Temporal IoU:** $\ge 0.30$
- **Onset Tolerance Window:** $\le 2.5$ seconds
- **Offset Tolerance Window:** $\le 3.5$ seconds
- **Prediction Deduplication:** 3.0s cooldown window merging consecutive triggers
