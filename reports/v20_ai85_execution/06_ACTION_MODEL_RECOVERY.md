# ROADWATCH V20 — Action Model Recovery
## 1. Architecture Mismatch
- Historical V5 ResNet18-GRU and V6 Hierarchical checkpoints are deep PyTorch models.
- The `OnlineE3` interface in `training/dmd/online_candidate.py` requires a JSON-serialized `NumericHead` (StandardScaler + OvR linear weights).
- No fitted `NumericHead` exists locally. Replacing deep models with dummy linear weights without authorized training data is prohibited.
