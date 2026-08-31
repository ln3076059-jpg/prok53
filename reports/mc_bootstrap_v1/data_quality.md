# MC bootstrap v1 data quality

Status: **PASS**

Dataset status: `PROPOSAL_MODEL_ONLY_NOT_GOVERNED_FINAL_DATA`

Manifest samples: 18099
Near pairs at Hamming ≤ 4: 118786
Cross-split near pairs: 0

## Splits

- `train`: {'images': 14581, 'instances': {'phone': 2094, 'seatbelt_fastened': 5963, 'seatbelt_unfastened': 1219}, 'positive_images': {'phone': 1857, 'seatbelt_fastened': 5903, 'seatbelt_unfastened': 1216}}
- `val`: {'images': 1782, 'instances': {'phone': 229, 'seatbelt_fastened': 361, 'seatbelt_unfastened': 284}, 'positive_images': {'phone': 181, 'seatbelt_fastened': 359, 'seatbelt_unfastened': 284}}
- `test`: {'images': 1736, 'instances': {'phone': 225, 'seatbelt_fastened': 322, 'seatbelt_unfastened': 302}, 'positive_images': {'phone': 177, 'seatbelt_fastened': 320, 'seatbelt_unfastened': 301}}

## Errors

- None

PASS validates file integrity and leakage isolation only. All labels remain proposal-only and require human review before final MC_001 training or accuracy claims.
