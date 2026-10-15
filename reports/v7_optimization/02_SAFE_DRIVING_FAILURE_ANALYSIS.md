# SAFE driving failure analysis

All values below come from outer-LOSO held-out predictions; they were not used to tune thresholds.

- Pooled SAFE support: 668
- Correct SAFE: 84
- SAFE→PHONE: 196
- SAFE→OTHER: 388

Per-subject detail, probability quantiles, phone-versus-negative margins, fine-head distributions, and temporal runs are preserved in `v7_safe_failure_analysis.json`. The result supports testing coarse-objective and binary-gate changes before any fine-head expansion.
