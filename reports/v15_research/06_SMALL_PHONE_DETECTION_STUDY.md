# Small-phone detection study

V15 adds `training/dmd/sliced_phone.py`, a SAHI-inspired adapter around the existing `PhoneDetector` protocol. It covers an original-resolution ROI with overlapping tiles, converts tile-local XYXY boxes to source-frame coordinates, preserves checkpoint hash/confidence/source, and runs confidence-ordered single-class NMS. It accepts no ground truth or review metadata.

Synthetic 1280×720 geometry on ROI `[0,72,832,684]` with 384×384 tiles and 25% overlap produced 6 tiles. Thirty adapter-only runs measured p50/p95/p99 0.186/0.308/0.464 ms. Model inference was excluded. Tests verify boundary coverage, local-to-original mapping, overlap deduplication, empty output and invalid config.

Decision: KEEP engineering; accuracy BLOCKED. No complete positive/negative bbox benchmark exists, so phone AP50, mAP50-95, precision, recall and negative-frame FP remain null. The adapter is not connected to active production config.
