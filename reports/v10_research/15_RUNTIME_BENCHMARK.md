# V10 runtime benchmark

Evidence: inherited measured local CPU benchmark on the same AMD Ryzen 5 5500U class host, no CUDA.

| Component | p50 | p95 | Throughput/scope |
| --- | --- | --- | --- |
| Detector | 520.11 ms | 1375.01 ms | 24 frames |
| Temporal head on cached embeddings | 0.272 ms | 0.416 ms | 2180.32 clips/s |
| Full backend context | 600.79 ms | 2050.07 ms | 1.119 FPS |

Full pipeline memory peaked at 631.0 MiB. Detector, ROI, encoder, temporal,
fusion, and seatbelt were not all activated in one end-to-end call, so missing stage latencies remain **NOT
MEASURED**. Cached-feature throughput is not full-system FPS. GPU latency and memory are not reported because
no NVIDIA runtime is available.
