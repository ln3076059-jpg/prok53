# 12. Temporal Localization & ActionFormer Recovery

## 1. ActionFormer Principles
ActionFormer utilizes a multi-scale feature pyramid with local self-attention to predict action boundaries $(s_i, e_i)$ directly from video feature sequences without rigid sliding window constraints.

## 2. Parameter Tuning in TemporalEventEngine
To recover the 5 short-duration FN events identified in root-cause analysis:
* `min_positive_seconds`: Re-calibrated from 1.20s down to **0.50s** (active in `model_config_v4_candidate.yaml`).
* `min_observations`: Re-calibrated from 3 down to **2** observations.
* `cooldown_seconds`: 3.0s to prevent alert fragmentation during burst interactions.
* `occlusion_bridge_seconds`: 2.0s to maintain continuity when phone passes behind the steering wheel.
