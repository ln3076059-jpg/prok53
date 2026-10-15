# Real Inference Performance Benchmark

Status: `MEASURED_LOCAL_CPU` on 6 physical / 12 logical CPUs and 6006 MB RAM. CUDA available: `False`; peak VRAM 0.0 MB.

| Component | Throughput / mean | p50 | p95 |
|---|---:|---:|---:|
| Video decode | 65.839 FPS | 14.846 ms | 16.703 ms |
| YOLO detector calls | 489.591 ms | 426.099 ms | 494.264 ms |
| Seatbelt classifier direct probe | 78.072 ms | 55.343 ms | 92.659 ms |
| Fusion direct | 0.0054 ms | 0.0056 ms | 0.0058 ms |
| Evidence JPEG write | 13.665 ms | 8.009 ms | 36.116 ms |
| ResNet embedding | 118.744 FPS | — | — |
| Temporal head | 2685.715 clips/s | 0.254 ms | 0.416 ms |
| Full backend context | 1.833 FPS overall | 478.431 ms | 538.685 ms |

Full pipeline processed 60 frames in 32.725s (real-time factor 16.232× against source FPS 29.760). RSS was 311.1 MB → peak 728.7 MB → end 554.5 MB.

This is an actual local cabin-video pipeline benchmark, not roadside accuracy evidence. Cached temporal-head clips/s is deliberately separated from full-pipeline FPS. Real-time capability is claimed only if full processing FPS meets the target stream rate; otherwise it does not.
