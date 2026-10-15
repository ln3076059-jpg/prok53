# Runtime benchmark

V7 temporal-head median throughput across measured folds is 2841.0 clips/s (range 267.4–3815.2) on local CPU. The unchanged full backend benchmark remains 1.833 FPS with detector mean 489.59 ms and peak RSS 728.7 MB.

The detector, not the temporal head, is the measured bottleneck. No detector cadence or input resolution was reduced because there is no independent short-event accuracy set to quantify the miss risk. This is a deliberate no-change performance decision, not an optimization claim.
