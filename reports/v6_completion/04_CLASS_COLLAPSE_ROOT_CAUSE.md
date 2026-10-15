# Class-Collapse Root-Cause Analysis

## Observed data

| Subject | Frames | SAFE | TEXT_L | TEXT_R | CALL_L | CALL_R | OTHER | Mean luminance |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| gB-9 | 809 | 116 | 30 | 28 | 85 | 66 | 76 | 0.3348 |
| gZ-36 | 1024 | 119 | 41 | 45 | 110 | 109 | 84 | 0.3314 |
| gC-14 | 795 | 105 | 24 | 26 | 75 | 82 | 82 | 0.2901 |

Adjacent training clips share 7 of 9 frames (77.78%). This is expected temporal overlap, but it reduces effective sample diversity.

## Supported cause

V5 B3 combined replacement sampling with square-root inverse-frequency loss weights. Their joint effect approximately equalized gradient mass twice. SAFE is not rare in the observed clips, so this can suppress its natural prior and encourage positive-class predictions. V5's SAFE recall of 0% is therefore consistent with an over-aggressive balancing objective. The V6 ablation directly tests unweighted, loss-only, sampler-only and hierarchical alternatives.

Preprocessing was consistent: frozen ImageNet ResNet-18, 64→112 resize, ImageNet normalization, cached embeddings for both train and inference, and center-frame labels. Class index mappings were verified. Exact duplicated embeddings and subject centroid distances are recorded in `v6_class_collapse_ablation.json`.

## Remaining hypotheses

- Subject/camera shift is plausible only where luminance, RGB motion and embedding-centroid evidence differ; it is not asserted from appearance alone.
- Training accuracy reaching 100% in several folds is evidence of limited effective diversity, not proof of generalization.
- Action confidence becoming one long event can cause temporal overmerge and matching penalties even when the interval overlaps real phone-use truth.
