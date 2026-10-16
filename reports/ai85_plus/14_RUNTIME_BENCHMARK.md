# Measured runtime and scope

Actual current development backend VideoAnalyzer pipeline: 24 frames, 21.4421 seconds, **1.119 processing FPS**, with SQLite and evidence path. Detection rows: 0; event rows: 0. This short cabin clip is a smoke benchmark and is not roadside accuracy evidence.
Detector/context p50/p95, decode, encoder, cached temporal head, seatbelt direct probes, fusion direct probes, evidence writes, RAM and CUDA/VRAM are in [runtime_benchmark.json](runtime_benchmark.json). Zero in-pipeline calls mean that component was unexercised; separately timed direct probes do not establish loaded event-path throughput.
Adaptive research extraction:

| Subject | Dense samples | Extraction seconds | Peak RSS MiB |
| --- | --- | --- | --- |
| gB-9 | 4043 | 111.04 | 543.0 |
| gZ-36 | 5117 | 136.651 | 525.2 |
| gC-14 | 3972 | 108.965 | 404.0 |

Dense extraction uses batched frozen features and shared contexts. It is not full-system FPS. T1 uses about 5× the action centers; unique required context embeddings are recorded separately per subject and sampler. A real scheduler with context reuse would be needed to realize savings. No integrated adaptive full-pipeline speed claim is made because no eligible adaptive model was promoted.
Spatial TTA/resolution/ROI timings are measured in phone_spatial_probes.json; no universal speed/quality gain is asserted. CPU benchmarks include local process load and have small-sample limits.

Private Kaggle job [longngoc11/roadwatch-ai85-plus-frozen-gpu-v8](https://www.kaggle.com/code/longngoc11/roadwatch-ai85-plus-frozen-gpu-v8): COMPLETE. It runs frozen phone inference on 64 legacy validation images already in the account. No DMD media, holdouts or credentials were uploaded.

| GPU | Variant | Images | Detector-call FPS | p50 ms | p95 ms |
| --- | --- | --- | --- | --- | --- |
| Tesla T4 | original_960 | 64 | 21.44 | 34.12 | 76.3 |
| Tesla T4 | resolution_640 | 64 | 38.47 | 15.29 | 67.08 |
| Tesla T4 | TTA_640 | 64 | 19.89 | 35.12 | 128.8 |
| Tesla T4 | resolution_1024 | 64 | 28.84 | 27.46 | 70.37 |
| Tesla T4 | agnostic_640 | 64 | 66.24 | 14.88 | 17.02 |

GPU and local CPU probes use different image sets; their throughput is not a controlled hardware speedup comparison. GPU detector-call FPS is not full-system FPS, and no AP/accuracy target was measured on Kaggle.
