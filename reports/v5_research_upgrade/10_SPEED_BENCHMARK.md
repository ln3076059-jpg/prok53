# Benchmark tốc độ V5 action branch

Environment: Windows, Python 3.11.9, PyTorch 2.13.0+cpu, device CPU,
CUDA=False; OMP/BLAS/MKL threads cố định 1 để tránh paging/thread oversubscription.

## B2 TinyTemporalCNN trên RGB cache

| Heldout subject | Clips/s | ms/clip total | p50 model ms | p95 model ms | Peak process RSS MB |
|---|---:|---:|---:|---:|---:|
| gB-9 | 1542.458 | 0.648 | 0.376 | 0.488 | 274.570 |
| gZ-36 | 1596.397 | 0.626 | 0.356 | 0.440 | 261.738 |
| gC-14 | 1562.677 | 0.640 | 0.367 | 0.535 | 262.727 |

Aggregate mean=1567.177 clips/s, p50=0.367ms,
max-fold p95=0.535ms, peak RSS=274.570MB.
RSS được sample mỗi 10ms và bao gồm Python/PyTorch + model + subject caches; không phải riêng model.
VRAM=`N/A` vì CUDA unavailable. Đây chỉ là action branch trên cache 64×64, không gồm video decode/ROI
extraction hoặc full YOLO/pose pipeline, nên không được gọi là full-system FPS.

## B3 temporal head trên ResNet-18 embeddings

| Heldout subject | Clips/s | ms/clip total | p50 model ms | p95 model ms | Peak process RSS MB |
|---|---:|---:|---:|---:|---:|
| gB-9 | 6086.548 | 0.164 | 0.131 | 0.166 | 320.859 |
| gZ-36 | 5966.135 | 0.168 | 0.128 | 0.181 | 334.117 |
| gC-14 | 4205.931 | 0.238 | 0.172 | 0.306 | 345.062 |

Aggregate temporal-head mean=5419.538 clips/s,
p50=0.131ms, max-fold
p95=0.306ms, peak RSS=345.062MB.

| Subject | Frames embedded | One-time extraction seconds |
|---|---:|---:|
| gB-9 | N/A | REUSED |
| gZ-36 | N/A | REUSED |
| gC-14 | N/A | REUSED |

Số B3 clips/s chỉ đo BiGRU temporal head trên embedding đã tính sẵn. Thời gian ResNet-18 extraction được
tách riêng ở bảng trên; do đó không được gọi là end-to-end throughput hay so trực tiếp với B2 clips/s.
Full-pipeline FPS=`NOT_MEASURED`; không nội suy từ cached clips/s.
