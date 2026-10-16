# Action representation and confidence diagnostics

The hierarchical ResNet18/BiGRU checkpoints were reused, with the same six-class map and physical temporal context. Spatial input remains 64px downsample then bilinear 112px: this experiment isolates sampling. Mean/attention/native112 training failures were not repeated.
The earlier subject-normalized direct coarse probe has SAFE recall 70.81%, separately from the binary branch event F1 65.31%. Improving SAFE argmax does not ensure sustained PHONE probability crosses the inherited 0.80 event threshold. Coarse PHONE sums four fine probabilities; fine SAFE argmax is a different decision.
Prior representation diagnostics: {'silhouette_cosine_by_action': 0.04861282929778099, 'silhouette_cosine_by_subject': 0.4095304608345032, 'subject_identification_accuracy_random_split': 1.0, 'interpretation': 'Subject identity is substantially more separable than action in frozen embeddings.'}. Whole-video normalization uses future unlabeled statistics and remains an offline transductive experiment.

| Variant | Six-class Macro-F1 | Clip accuracy | Coarse SAFE recall | Evaluated centers |
| --- | --- | --- | --- | --- |
| T0 | 34.23% | 36.75% | 12.57% | 2604 |
| T1 | 34.41% | 36.85% | 12.36% | 12972 |
| T2 | 34.49% | 40.58% | 12.61% | 3433 |
| T3 | 34.38% | 36.77% | 12.43% | 7789 |
| T4 | 33.84% | 41.70% | 11.85% | 3964 |
| T5 | 33.84% | 41.70% | 11.85% | 3964 |

Different samplers evaluate different center distributions; these clip metrics are descriptive and not a common-grid classifier ranking. Full per-class confusion matrices remain in temporal_results.json. No new classifier was trained without approved labels and a legitimate independent validation split. Behavior ROI and temporal dynamics informed implemented probes, not paper reproduction. [Primary sources](references.md).
