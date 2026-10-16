# Measured local performance

| Variant | Subject | Head inference seconds | Cached clips/s |
| --- | --- | --- | --- |
| MEAN_FINETUNE | gB-9 | 0.259178 | 3090.542 |
| MEAN_FINETUNE | gZ-36 | 0.365305 | 2781.241 |
| MEAN_FINETUNE | gC-14 | 0.241661 | 3256.629 |
| ATTENTION_FINETUNE | gB-9 | 0.251921 | 3179.567 |
| ATTENTION_FINETUNE | gZ-36 | 0.336346 | 3020.701 |
| ATTENTION_FINETUNE | gC-14 | 0.276049 | 2850.943 |
| NATIVE112_FINETUNE | gB-9 | 0.260514 | 3074.685 |
| NATIVE112_FINETUNE | gZ-36 | 0.327068 | 3106.385 |
| NATIVE112_FINETUNE | gC-14 | 0.281297 | 2797.758 |

Cached-head throughput excludes decoding and ResNet inference. Native112 extraction metadata records complete video-decoding-plus-embedding time per subject. CPU, PyTorch, memory and disk snapshots are in `environment_audit.json`. CUDA is unavailable.

```json
{
  "frames": 12,
  "source_fps": 29.76,
  "processing_seconds": 8.2044,
  "processing_fps": 1.463,
  "real_time_factor": 20.347,
  "per_context_latency": {
    "count": 12,
    "mean_ms": 671.3897,
    "p50_ms": 457.4039,
    "p95_ms": 1714.1354
  },
  "database_detection_rows": 0,
  "database_event_rows": 0,
  "scope": "ACTUAL VideoAnalyzer + SQLite + evidence path on a cabin clip"
}
```

This is a short real VideoAnalyzer + SQLite/evidence-path smoke benchmark. It measures the active V2 backend, not a deployed research action branch. No research checkpoint was substituted into the backend. The 12-frame clip produced zero detection/event rows and did not exercise classifier or fusion calls within the pipeline. Their direct probes are separate measurements in the JSON artifact.
