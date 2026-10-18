# Phone detection results

The V15 tiled implementation was verified at code and regression-test level: tile bounds,
overlap, source-coordinate mapping, edge handling, duplicate suppression, ROI clipping and
unchanged checkpoint identity are covered. D0-D4 accuracy ablations were not run because there
is no approved validation set containing human-checked phone boxes and reviewed in-scope
negative frames.

Phone AP50, AP50-95, small-object recall and negative-frame FP rate remain **not measured**.
The 13 historical FN-selected frames were not reused as the principal detector validation set.
Proposal yield was not reported as recall. No detector was promoted and no active weights or
thresholds changed.
