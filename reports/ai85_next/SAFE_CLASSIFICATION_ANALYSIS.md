# SAFE_DRIVING classification analysis

## Baseline

The baseline has 668 SAFE clips, so SAFE is not rare. Only 84 are classified SAFE; 196 are
classified PHONE and 388 OTHER.

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

| Subject | SAFE frames | PHONE frames | OTHER frames |
| --- | ---: | ---: | ---: |
| gB-9 | 240 | 416 | 152 |
| gZ-36 | 241 | 607 | 175 |
| gC-14 | 208 | 416 | 170 |

## Controlled representation and head comparison

| Model | SAFE recall | Coarse Macro-F1 | Balanced accuracy |
| --- | ---: | ---: | ---: |
| V6 baseline hierarchical head | 12.57% | 46.48% | 50.99% |
| Subject-normalized six-class probe, regrouped | 0.45% | 24.37% | 33.62% |
| Subject-normalized direct coarse probe | 70.81% | 63.07% | 63.41% |

Per-video z-score normalization uses no labels and reduces static appearance offsets. The direct
coarse probe improves SAFE recall by
`58.24%` absolute, but remains below 85%.

The six-class result demonstrates why flat-class accuracy cannot substitute for official coarse
SAFE recall: summing four individually modest PHONE probabilities can reverse the decision. The
direct coarse head is the valid comparison for SAFE.

## Preprocessing and label checks

The prior audit verified exact alignment between RGB caches and embeddings and found zero replay
label mismatches with the stored float32 timestamps. There are 3-6 boundary disagreements per
subject only when timestamps are widened to float64. Nine-sample windows are mixed near action
boundaries, but long SAFE intervals also fail, so boundary mixing is not the primary explanation.
