# Training ablations

| Variant | Clip accuracy | Six-class macro-F1 | Coarse macro-F1 | SAFE recall | Phone recall | OTHER recall |
|---|---:|---:|---:|---:|---:|---:|
| V7_A_V6_BASELINE | 36.75% | 34.23% | 46.48% | 12.57% | 77.21% | 63.18% |
| V7_B_COARSE_OBJECTIVE | 36.44% | 31.82% | 40.95% | 0.15% | 94.79% | 47.08% |
| V7_C_HARD_NEGATIVES | 37.25% | 34.21% | 41.92% | 2.54% | 78.25% | 70.02% |
| V7_D_BINARY_GATE | 37.90% | 33.32% | 41.41% | 1.05% | 88.53% | 56.14% |
| V7_E_LONGER_CONTEXT | 32.98% | 29.58% | 34.99% | 0.00% | 99.24% | 22.33% |
| V7_F_NEGATIVE_MARGIN | 36.37% | 31.67% | 40.74% | 0.15% | 90.83% | 52.11% |

Training seconds across three folds: `{"V7_A_V6_BASELINE": 121.896, "V7_B_COARSE_OBJECTIVE": 97.09, "V7_C_HARD_NEGATIVES": 115.72, "V7_D_BINARY_GATE": 102.763, "V7_E_LONGER_CONTEXT": 93.938, "V7_F_NEGATIVE_MARGIN": 97.09}`. V7-B changes only coarse/fine loss treatment; V7-C adds verified training-only hard-negative weighting; V7-D changes to a binary PHONE/NON_PHONE gate; V7-E changes only sequence length from 9 to 17; V7-F applies the pre-registered 0.15 negative-evidence margin to V7-B and keeps suppressed clips in every denominator.
