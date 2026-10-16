# Root-cause analysis

## Finding

The dominant measured failure is cross-subject representation shift, followed by a lack of
continuous PHONE evidence. The frozen ResNet18 embeddings separate subject identity much more
strongly than behavior: subject silhouette is `0.4095` versus
action silhouette `0.0486`. A linear probe identifies the
subject with `100.00%` accuracy on a
random split.

This conclusion uses embedding separability, training/outer performance, image statistics,
class distributions, temporal traces, and direct head ablations. It is not inferred from a
confusion matrix alone.

## Memorization evidence

| Fold | Final epoch | Training coarse accuracy | Loss |
| --- | ---: | ---: | ---: |
| Held out gB-9 | 25 | 100.00% | 0.290047 |
| Held out gZ-36 | 25 | 100.00% | 0.284921 |
| Held out gC-14 | 25 | 100.00% | 0.286323 |

| Outer subject | Six-class accuracy | SAFE recall |
| --- | ---: | ---: |
| gB-9 | 47.32% | 27.90% |
| gZ-36 | 33.46% | 4.70% |
| gC-14 | 30.24% | 3.98% |

The near-perfect training accuracy and poor subject-disjoint performance are consistent with
memorization of subject/camera appearance. Exact RGB duplicate checks from the prior audit found
no cross-subject duplicate frames, so duplication does not explain the gap.

## Head-design evidence

The subject-normalized six-class probe reaches Macro-F1 `56.54%`, but its
official coarse SAFE recall is only
`0.45%` because the four PHONE class
probabilities are summed before the coarse decision. A direct three-class head avoids this
aggregation failure and reaches SAFE recall
`70.81%`, coarse Macro-F1
`63.07%`, and balanced accuracy `63.41%`.
It still misses the 85% SAFE target.

## Event-loss attribution

- `11` of 13 baseline FNs lack two
  consecutive samples at the inherited 0.80 threshold.
- `1` reaches sustained activation but fails
  boundary/matching.
- `1` has no centered action sample.
- The 960-pixel detector trace produces a PHONE proposal in only
  `2/13` FN intervals and
  `7/52` sampled frames.

## Limits

Only three already-consumed development subjects are available. The source governance record
states `RIGHTS_STATUS = PENDING_PROJECT_REVIEW` and `RIGHTS_ACCEPTED = false`. No causal claim
about lighting or occlusion is made because those factors have no verified case-level labels.
