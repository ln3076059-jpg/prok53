# V9 representation experiments

No model was trained. Existing frozen experiments and new ROI/motion instrumentation were analyzed first.

- Subject silhouette: 0.4095.
- Action silhouette: 0.0486.
- Subject/action separability ratio: 8.42×.
- Dense result: One long event recovered; short-event matching remained 0/4.
- CLIP comparison: `BLOCKED_NO_FROZEN_CLIP_ARTIFACT_IN_REPOSITORY`.

Frozen features retain much more subject identity than action structure. The next legitimate experiment is
a hash-registered comparison of full ROI, face/hand/torso crops, RGB differences and CLIP features under
the same LOSO split, reporting clip Macro-F1 separately from event F1.
