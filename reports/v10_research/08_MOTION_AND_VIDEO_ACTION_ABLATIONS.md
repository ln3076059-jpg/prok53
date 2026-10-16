# V10 motion and video-action ablations

The controlled CPU probe holds LOSO folds and the balanced linear head fixed. Temporal residual D2 raises
pooled clip Macro-F1 from 17.56% to 51.54% and
SAFE recall from 6.82% to 44.27%.
Subject silhouette falls from 0.4014 to
0.0895, but action silhouette also falls from
0.0278 to
0.0046.

The worst D2 fold is `gZ-36`: Macro-F1 37.64%, SAFE recall 1.66%. D2 also uses a future adjacent frame, so it
is an offline non-causal representation. Event F1, onset latency, short-event recall, fragmentation, optical
flow, D3 CLIP, and D4 recurrent-head results are **NOT MEASURED** in V10. Downloads and larger training were
not justified on a no-CUDA host with unaccepted dataset rights. Decision: retain D2 only as a development
hypothesis; reject promotion.
