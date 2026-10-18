# 10. Action Model & Fusion Audit
- **Fitted E3 Head:** Retained as `null` in manifest because historical E3 was evaluated as per-subject LOSO models rather than a single unified exported pipeline artifact.
- **Fusion Logic:** `fused_score = min(raw_phone_score, action_score["PHONE_USE"])`.
- **Normalization:** Enforced `HANDHELD_CONTEXTS = {"HANDHELD", "HANDHELD_USE", "HANDHELD_PHONE_USE"}` to prevent valid driver phone interactions from being dropped at fusion stage.
