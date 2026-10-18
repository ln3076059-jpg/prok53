# Seatbelt study

The inherited best development Macro-F1 is 0.5852; UNCERTAIN recall remains 0.04 in all reported variants that retain any uncertain predictions. The best historical S2 flip-TTA result has balanced accuracy 0.58396, false-unfastened rate 0.08 and abstention rate 0.054602.

The evidence supports a future two-stage model: first classify `OBSERVABLE` versus `UNCERTAIN_OR_OCCLUDED`; only then classify observable torso crops as FASTENED/UNFASTENED. RST-Net motivates torso ROI, pruning and Triplet Attention, while the overhead YOLOv7 paper has a material camera-domain mismatch.

All 500 review candidates remain pending. V15 did not approve labels, train, calibrate or change seatbelt runtime. Invisible belts are never UNFASTENED. Decision: BLOCKED pending authenticated human labels, rights and a group-disjoint split.
