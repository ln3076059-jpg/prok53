# Controlled action-model experiments

Reused the existing frozen ImageNet ResNet18 and V6 BiGRU hierarchy. Each continuation starts from the checkpoint that excluded the same outer subject. Training uses only the other two subjects; outer embeddings are loaded after optimization. Five epochs, batch 16, learning rate 0.0001, sqrt-inverse weighted CE, no replacement sampling. AdamW is new: weight continuation rather than exact optimizer resume.

MEAN_FINETUNE controls additional optimization. ATTENTION_FINETUNE replaces only mean pooling with zero-initialized learned temporal attention; initial outputs match the inherited model. NATIVE112_FINETUNE changes spatial sampling only, with mean pooling and identical continuation settings.

| Variant | Outer subject | Seed | Train samples | Train seconds |
| --- | --- | --- | --- | --- |
| MEAN_FINETUNE | gB-9 | 8516 | 902 | 7.279 |
| MEAN_FINETUNE | gZ-36 | 8517 | 795 | 6.15 |
| MEAN_FINETUNE | gC-14 | 8518 | 909 | 7.178 |
| ATTENTION_FINETUNE | gB-9 | 8516 | 902 | 7.079 |
| ATTENTION_FINETUNE | gZ-36 | 8517 | 795 | 6.306 |
| ATTENTION_FINETUNE | gC-14 | 8518 | 909 | 7.166 |
| NATIVE112_FINETUNE | gB-9 | 8516 | 902 | 7.331 |
| NATIVE112_FINETUNE | gZ-36 | 8517 | 795 | 6.031 |
| NATIVE112_FINETUNE | gC-14 | 8518 | 909 | 7.174 |

Final epoch was fixed before evaluation. No outer-label early stopping or threshold search was performed. Checkpoint/prediction/input/config/source hashes and complete training histories are retained with each fold. Attention failure is retained. See `references.md` for primary research sources and applicability limits.
