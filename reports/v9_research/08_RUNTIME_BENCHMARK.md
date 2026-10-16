# V9 runtime benchmark

| Frozen T4 variant | Images | Detector FPS | p50 ms | p95 ms |
| --- | --- | --- | --- | --- |
| original_960 | 64 | 21.44 | 34.12 | 76.30 |
| resolution_640 | 64 | 38.47 | 15.29 | 67.08 |
| TTA_640 | 64 | 19.89 | 35.12 | 128.80 |
| resolution_1024 | 64 | 28.84 | 27.46 | 70.37 |
| agnostic_640 | 64 | 66.24 | 14.88 | 17.02 |

The probe used one Tesla T4 device (`device=0`) and 64 legacy validation images without labels. It measures
detector calls, not full-system FPS. Local full-backend processing is 1.119 FPS on different inputs/hardware,
so no CPU/GPU speedup is claimed. V9 diagnostics add offline analysis only.
