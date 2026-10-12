# Roadwatch V4 Experiment Plan

## Fixed governance

- Development subjects: `gB-9`, `gZ-36`, `gC-14` only.
- Consumed holdouts forbidden for development: `gZ-37`, `gE-28`.
- Baseline and candidate must use identical annotations, subject pool and event matching policy.
- A new independent holdout is evaluated once only after configuration selection and provenance lock.
- Selection rule: maximize development F1 subject to false alarms/min <= 0.30, then prefer higher
  recall, lower onset latency and lower compute cost in that order.

## Ablations

1. Historical V3 baseline.
2. BODY-primary gated fusion.
3. Pose-conditioned FACE/HANDS rescue.
4. Reliability-aware view weighting.
5. Six-state phone context with strict abstention.
6. Stable-occupant temporal association and 2 s occlusion bridge.
7. Combined V4 candidate.

Each run must retain immutable config, code commit, input hashes, predictions, TP/FP/FN matching,
latency and resource measurements. Detector thresholds remain inherited until development evidence
justifies calibration; holdout data must never select them.

## Current execution result

`training/dmd/run_v4_experiments.py` was run. It returned `BLOCKED` because all three required
development feature caches are absent. Machine-readable details are in `experiment_preflight.json`.
No ablation was substituted from historical aggregates.

## Reproducible command

```powershell
py -3.11 training/dmd/run_v4_experiments.py
```
