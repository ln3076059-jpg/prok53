# Roadwatch V27 AI85 execution package

Private code-only package for V27 regression checks and the V30 causal-action
contracts. V30 model fitting remains local-only because DMD artifacts are excluded.

No DMD video, frame, annotation, embedding, action checkpoint, raw prediction,
or credential is included. The cached-development reference is TP/FP/FN = 12/13/8,
precision = 0.48, recall = 0.60, and F1 = 0.533333.

This package supports code verification and GPU preflight only. It cannot produce
raw DMD event metrics unless an independently authorized input dataset is attached.
The included phone-component result is provisional: its bootstrap labels are model
proposals, not governed final ground truth, and it is not an event-level metric.
The included aggregate raw-development result is 14/3/6 TP/FP/FN with precision
0.8235, recall 0.7000, and F1 0.7568; the >0.85 gate remains unmet.
