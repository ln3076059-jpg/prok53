# ROADWATCH V19 — Action Model Recovery & OnlineE3 Status
## 1. Status of Action Checkpoints
- Historical V5 ResNet18-GRU and V6 Hierarchical checkpoints exist in `models/recovery_extracted/`.
- However, the fitted JSON `NumericHead` required for real-time `OnlineE3` streaming inference is absent from the active filesystem.
- Synthetic dummy heads were used in tests; per V19 engineering rules, synthetic weights cannot be reported as accuracy measurements.
