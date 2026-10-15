# Subject Generalization Analysis

| Subject | Luminance mean | Luminance std | Median RGB Δ | Median adjacent embedding cosine |
|---|---:|---:|---:|---:|
| gB-9 | 0.3348 | 0.2779 | 0.0322 | 0.9774 |
| gZ-36 | 0.3314 | 0.2638 | 0.0225 | 0.9721 |
| gC-14 | 0.2901 | 0.2668 | 0.0318 | 0.9545 |

V5 B8 produced five of seven false positives on gZ-36 (historical FA/min ≈0.5816). Inspection of event intervals shows that at least one long unmatched prediction overlaps multiple ground-truth phone events; this is categorized as temporal overmerge/matching mismatch rather than automatically as safe-driving confusion.

Camera angle, clothing, phone position and posture were not directly labeled, so they remain hypotheses. The quantitative audit supports subject-dependent image/embedding differences but cannot assign causal semantics to them. Only three development subjects exist, making nested subject validation high variance.

V6 uses outer LOSO and one fixed pre-registered threshold. It does not claim an unbiased final estimate because architecture selection still observes the same three development subjects.
