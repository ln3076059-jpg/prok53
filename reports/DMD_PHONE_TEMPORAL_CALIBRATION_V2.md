# DMD Phone Temporal Calibration Report

**Status:** COMPLETE  
**Dataset Role:** TEMPORAL_DEVELOPMENT (Non-Canonical, Development Only)  
**Calibration Subjects:** 14, 36  
**Phone Model SHA256:** `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`  
**Selected Config Hash:** `bd3cdc8b481312af9b8248e75dfe9a2a77640c9feb41d86234cd3a9d0465fd5e`  
**Selection Criterion:** Highest F1 with false alarms / min $\le 1.0$  

---

## 1. Candidate Comparison Grid

| Configuration | Window (s) | Min Pos (s) | Pos Ratio | Precision | Recall | F1 | False Alarms / min | Start Latency (s) |
|---|---|---|---|---|---|---|---|---|
| `baseline_001_no_bridge` | 1.50 | 0.50 | 0.50 | 21.1% | 20.0% | 20.5% | 0.98 | +22.67s |
| `fast_bridge_25` | 1.50 | 0.40 | 0.40 | 54.5% | 30.0% | 38.7% | 0.33 | +17.36s |
| `balanced_bridge_35` | 2.00 | 0.50 | 0.45 | 60.0% | 30.0% | 40.0% | 0.26 | +16.43s |
| `deep_bridge_40` **(SELECTED)** | 2.00 | 0.50 | 0.40 | 66.7% | 30.0% | 41.4% | 0.20 | +15.51s |
| `conservative_bridge_30` | 2.50 | 0.80 | 0.50 | 50.0% | 25.0% | 33.3% | 0.33 | +11.72s |

---

## 2. Selected Frozen Calibration Parameters

```json
{
  "name": "deep_bridge_40",
  "window_seconds": 2.0,
  "min_positive_seconds": 0.5,
  "min_observations": 2,
  "positive_ratio": 0.4,
  "cooldown_seconds": 3.0,
  "gap_tolerance_seconds": 3.0,
  "feature_positive_score": 0.18,
  "candidate_threshold": 0.18,
  "activation_threshold": 0.25,
  "release_threshold": 0.15,
  "occlusion_bridge_seconds": 4.0
}
```

This configuration is permanently frozen for the held-subject benchmark.
