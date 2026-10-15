# Phone Experiment Plan

All variants use the same frozen ResNet-18 embeddings, sequence length 9, LOSO subjects, 25 epochs, batch 16, and fixed event threshold 0.80. Batch size was reduced from 64 before any complete V6 result after a documented local RAM allocation failure. No metric-driven threshold change was made.

| Variant | Architecture | Sampling | Loss weighting | Hypothesis |
|---|---|---|---|---|
| FLAT_UNWEIGHTED | flat | none | none | Preserve observed class priors and recover SAFE_DRIVING recall. |
| FLAT_LOSS_ONLY | flat | none | sqrt_inverse | Mild loss weighting can improve minority actions without replacement sampling. |
| FLAT_SAMPLER_ONLY | flat | sqrt_inverse | none | Replacement sampling alone is less aggressive than the V5 double-balanced objective. |
| HIERARCHICAL_LOSS_ONLY | hierarchical | none | sqrt_inverse | A SAFE/PHONE/OTHER gate reduces SAFE_DRIVING-to-phone confusion before fine action classification. |

Selection was registered before results: FA/min ≤0.30, worst-subject FA/min ≤0.60, SAFE recall ≥0.20, then maximum event F1 and recall. Stage 1 hierarchy contains SAFE/PHONE/OTHER; UNCERTAIN is an abstention policy, not a fabricated supervised label. Stage 2 predicts four phone subtypes only under PHONE mass.

Focal loss, balanced softmax, extra hard negatives and subject normalization were not run because the dataset has only three reusable subjects. Adding more knobs would increase development overfit risk.
