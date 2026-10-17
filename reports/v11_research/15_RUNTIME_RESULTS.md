# V11 runtime results

The inherited same-host CPU cabin-video benchmark reports detector p50 520.11 ms and p95 1375.01 ms over 24 frames. Full backend context latency is p50 600.79 ms and p95 2050.07 ms, with 1.119 processing FPS and real-time factor 26.588. Peak RSS was 631.0 MiB.

The V11 representation family took 37.47 seconds and the generalization family took 21.69 seconds on CPU, including all three LOSO folds in each planned variant. Fold-specific fit/predict times are stored with each prediction artifact.

Cached-feature throughput is not full-pipeline FPS. V11 did not rerun a new end-to-end detector+ROI+encoder+temporal+fusion pipeline, so incremental ROI and causal-motion latency inside production is NOT MEASURED. GPU latency and VRAM are not reported because no NVIDIA runtime is present.
