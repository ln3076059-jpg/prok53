# Dataset and preprocessing audit

| Subject | Cache frames | Views | Label mismatches | Float boundary diffs |
| --- | --- | --- | --- | --- |
| gB-9 | 809 | body, face, hands | 0 | 4 |
| gZ-36 | 1024 | body, face, hands | 0 | 3 |
| gC-14 | 795 | body | 0 | 6 |

RGB and embeddings align exactly. Cached labels reproduce against the original float32 timestamp labeling path. Widening timestamps to float64 creates 13 sub-frame boundary disagreements; these are numeric replay differences, not evidence of human label errors. No cached or original label was changed.

No exact RGB duplicate was found across the three subjects. Near duplicates were not measured. Overlapping windows exist within each subject; outer LOSO separates whole subjects before windows are built. Camera identity and driver identity cannot be disentangled with this pool. gC-14 lacks FACE/HANDS files.

The original action path shrinks ROIs to 64×64 then upsamples to 112×112. The new native112 extraction uses original BODY pixels with identical ROI bounds, frame IDs, labels, normalization and pretrained ResNet weights. Per-file hashes, class counts and checks are in `dataset_audit.json` and `native112/embeddings/*_extraction.json`. No new dataset acquisition was performed; original source/license restrictions remain.
