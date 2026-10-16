# Visual representation ablation

## Hypothesis

Frozen ImageNet embeddings contain a large subject/camera offset. Per-video label-free feature
normalization should suppress that offset before a larger temporal architecture is attempted.
A strongly regularized linear probe measures whether the normalized representation is separable.

## Results

| Representation/head | Clip or coarse metric | SAFE recall | Event F1 at inherited policy |
| --- | ---: | ---: | ---: |
| V6 frozen embedding + BiGRU | Six-class Macro-F1 34.23% | 12.57% | 64.00% |
| Normalized embedding + six-class linear | Six-class Macro-F1 56.54% | 0.45% | 6.67% |
| Normalized embedding + direct coarse linear | Coarse Macro-F1 63.07% | 70.81% | 0.00% |
| Normalized embedding + binary PHONE linear | PHONE event branch | n/a | 65.31% |

The representation/head change improves clip-level separability and SAFE recognition but does
not create sufficiently calibrated sustained event probabilities. The direct coarse probe emits
no event at the inherited 0.80 threshold because strong regularization keeps probabilities below
that threshold.

Mean fine-tuning, temporal attention and Native112 were not rerun. Multi-ROI training was not
performed: gC-14 lacks FACE/HANDS video, while the other two subjects' views differ from BODY by
7-22 frames. A three-subject group-disjoint multi-view comparison would therefore confound view
availability with subject identity.
