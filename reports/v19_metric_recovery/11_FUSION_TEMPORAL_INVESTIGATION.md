# ROADWATCH V19 — Fusion Logic & Temporal Engine Investigation
## 1. Temporal Config v2 Performance
`models/temporal_config_v2.json` (`deep_bridge_40`):
- `window_seconds`: 2.0s
- `min_positive_seconds`: 0.5s - 1.2s
- `occlusion_bridge_seconds`: 4.0s
- `cooldown_seconds`: 3.0s

The 4.0s occlusion bridge successfully bridges momentary hand drops during phone calls without creating duplicate alerts.
