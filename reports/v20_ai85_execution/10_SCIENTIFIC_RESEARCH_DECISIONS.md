# ROADWATCH V20 — Scientific Research Decisions
## 1. Key Engineering Adoptions
1. **Driver-ROI Native Slicing (inspired by SAHI, IEEE ICIP 2022):** Successfully deployed in `run_autonomous_v20.py` to preserve small phone detail.
2. **Causal Forward Hysteresis:** Occlusion bridging (4.0s) and cooldown deduplication (3.0s) without future-frame leakage.
3. **Strict Evaluation Stratification:** Clear separation of synthetic tests, diagnostic replays, autonomous video inference, and independent holdouts.
