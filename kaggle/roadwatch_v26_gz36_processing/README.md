# Roadwatch V26 gZ-36 processing package

Private Kaggle package containing reproducible processing code, frozen temporal policy,
regression tests, and aggregate V26 development metrics.

Excluded by design:
- source videos and decoded frames;
- OpenLABEL annotations and private labels;
- cached embeddings, model checkpoints, and raw predictions;
- Kaggle credentials or other secrets;
- independent or consumed holdout data.

The code expects authorized local inputs. The included aggregate results are development
diagnostics and do not establish >0.85 performance or production readiness.

Current verified cached result: best corrected run E2 has TP=12, FP=13, FN=8,
precision=0.48, recall=0.60, and F1=0.533333 on the frozen 20-event development cohort.
The corrected raw-video D2 diagnostic completed only gZ-36: TP=4, FP=4, FN=7,
precision=0.50, recall=0.363636, and F1=0.421053. gC-14 did not complete under the same
code, so this is a partial development diagnostic and not a two-subject benchmark.

The cached E2-E5 runs consume historical D5 proposals (specialist plus auxiliary), while
the raw diagnostic uses D2 (specialist only). These results are not a detector-controlled
parity comparison. The raw metric also predates the latest boundary collector, so no gain
from that change is claimed.
