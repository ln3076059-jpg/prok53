# DMD Phone Temporal Calibration Report

**Status:** COMPLETE  
**Dataset Role:** TEMPORAL_DEVELOPMENT (Non-Canonical, Development Only)  
**Calibration Subjects:** 14  
**Phone Model SHA256:** `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`  
**Selected Config Hash:** `5d70dc287251f79cdf3ecf5019f007cf80d4a7a4162bb49e1bf22e496549199c`  
**Selection Criterion:** Highest F1 with false alarms / min $\le 1.0$  

---

## 1. Candidate Comparison Grid

| Configuration | Window (s) | Min Pos (s) | Pos Ratio | Precision | Recall | F1 | False Alarms / min | Start Latency (s) |
|---|---|---|---|---|---|---|---|---|
| `window_15f_fast` **(SELECTED)** | 1.50 | 0.50 | 0.50 | 100.0% | 46.2% | 63.2% | 0.00 | +18.02s |
| `window_20f_balanced` | 2.00 | 0.60 | 0.50 | 100.0% | 20.0% | 33.3% | 0.00 | +12.21s |
| `window_30f_conservative` | 2.50 | 0.80 | 0.55 | 100.0% | 11.1% | 20.0% | 0.00 | +7.93s |
| `window_45f_strict` | 3.00 | 1.00 | 0.60 | 100.0% | 11.1% | 20.0% | 0.00 | +7.93s |

---

## 2. Selected Frozen Calibration Parameters

```json
{
  "name": "window_15f_fast",
  "window_seconds": 1.5,
  "min_positive_seconds": 0.5,
  "min_observations": 2,
  "positive_ratio": 0.5,
  "cooldown_seconds": 3.0,
  "gap_tolerance_seconds": 1.5,
  "feature_positive_score": 0.28,
  "candidate_threshold": 0.28,
  "activation_threshold": 0.32,
  "release_threshold": 0.2
}
```

This configuration is permanently frozen for the held-subject benchmark.
