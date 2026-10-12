# Roadwatch V4 Performance Benchmark

## Historical measured baseline

The checked-in `reports/runtime_benchmark.json` records a decoded-frame component benchmark on CPU:

- serial decode + component p50: 1254.30 ms
- p95: 1759.80 ms
- serial component throughput: 0.649 FPS
- behavior detector p50/p95: 1060.20/1564.80 ms
- pose p50/p95: 122.47/203.29 ms
- seatbelt classifier p50/p95: 49.53/97.48 ms
- fusion p50/p95: 0.0253/0.0369 ms
- CPU RAM: 233.09 MB

This excludes database writes, evidence encoding, queue delay and network transport. A separate V3
historical report records 0.9 processed FPS at 1060.84 ms per sampled multi-view frame.

## Current environment

PyTorch is CPU-only (`2.13.0+cpu`); CUDA and VRAM metrics are unavailable. Development videos are
missing, so end-to-end V4 inference was not benchmarked. No historical or synthetic number is
presented as V4 throughput.

GPU FP16, ONNX and TensorRT comparisons remain blocked pending compatible hardware, input media and
an accuracy-preserving development evaluation.
