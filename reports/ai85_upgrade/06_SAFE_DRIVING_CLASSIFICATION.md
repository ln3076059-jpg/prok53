# SAFE_DRIVING classification

| Variant | Coarse SAFE recall | Six-class macro-F1 |
| --- | --- | --- |
| V7_A_V6_BASELINE | 12.57% | 34.23% |
| MEAN_FINETUNE | 12.13% | 34.16% |
| ATTENTION_FINETUNE | 12.13% | 34.11% |
| NATIVE112_FINETUNE | 19.01% | 40.21% |

Baseline coarse confusion (rows/columns: SAFE, PHONE, OTHER):

```json
[
  [
    84,
    196,
    388
  ],
  [
    74,
    1111,
    254
  ],
  [
    9,
    174,
    314
  ]
]
```

Of 668 SAFE clips, 84 are classified SAFE, 196 PHONE and 388 OTHER. SAFE is not a rare class. Additional weighted-loss continuation and attention do not solve the failure. The large gap between near-perfect inherited training accuracy and outer performance supports a subject/domain generalization problem. This is an inference, not proof of the visual cause. All new confusion matrices and per-class supports are preserved in `combined_metrics.json`.
