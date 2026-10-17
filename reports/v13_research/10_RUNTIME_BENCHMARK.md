# V13 runtime benchmark

R0-R5 source-resolution extraction across one decoded frame has p50 25.939 ms, p95 29.556 ms and p99 31.173 ms. This timed the six ROI variants after frame decode; it is not detector, encoder, model, fusion, event persistence or full-pipeline latency.

The inherited E3 cached-feature timing remains roughly 3-7 ms, and the inherited full-backend benchmark remains 1.119 FPS with p50 600.79 ms and p95 2050.07 ms. V13 did not claim real-time operation, GPU FPS, end-to-end alert delay or full candidate throughput because the fitted candidate is absent.
