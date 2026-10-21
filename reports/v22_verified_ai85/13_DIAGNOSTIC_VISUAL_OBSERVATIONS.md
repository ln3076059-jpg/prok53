# Diagnostic visual observations

Scope: GT_INFORMED_DIAGNOSTIC. These are assistant observations of local source frames, not human label approval, a fitting dataset, or an autonomous benchmark. Source timestamps, frame indices, derived JPEG hashes and missing views are recorded for every case in the local `fp_fn_ledger.json`. BODY contact sheets are under `private/case_frames/` and are excluded from Git.

The five BODY sheets were inspected. The first inspection covered 29 cases; the ledger was then corrected to include the sixteenth FP, a duplicate-match penalty. Additional FACE/HANDS frames were extracted for gZ-36; their synchronization is unverified. This review does not assert temporal causes from a single frame.

| Cases | Visible observations and implication |
|---|---|
| FP_gC-14_01 | Midpoint shows a phone at the ear. Its interval overlaps an annotated phone event. The V21 narrative that this is a console phone is contradicted by the sampled visual evidence. Investigate short/fragmented alert matching, not an assumed console false detection. |
| FP_gC-14_02, _04, _05 | Sampled BODY frames show a handheld phone or phone near the ear. Some historical FP intervals overlap GT; a scoring FP is not necessarily a physically false phone proposal. |
| FP_gC-14_03, _06, _07, _08 | Sampled frames show a phone or rectangular handheld object around the wheel/lap. This does not independently prove use intent or the correctness of source interval boundaries. Do not relabel them using a contact sheet. |
| FN_gC-14_01 through _07 | Phone-like handheld devices or phone-to-ear posture are visible at the sampled midpoints. A general claim that these events are physically invisible is unsupported. All have phone proposals in at least one sampled frame in the fresh trace. |
| FP_gZ-36_01 through _04 | Sampled BODY frames show handheld objects near the wheel/lap. GT overlap and proposal/pose evidence must be considered separately. |
| FP_gZ-36_05 through _07 | Midpoints do not visibly establish handheld phone use. These images alone do not identify which detector box generated the alert or verify mounted-phone context. Root cause remains unresolved. |
| FP_gZ-36_08 | Added from repeated GT assignment in the frozen matcher, rather than from `unmatched_predictions`. Its additional original-resolution midpoint was inspected: a phone rests on the lap. Duplicate-scoring provenance is verified arithmetically; the precise action transition is not resolved by this image. This is why the original eight gZ-36 FP cannot be audited by reading only seven unmatched prediction records. |
| FN_gZ-36_01 through _07 | BODY midpoints show handheld-device or phone-to-ear evidence. Two intervals have no retained D5 proposals on the stride15 grid; other misses have proposals, so detector visibility alone cannot explain all FN. FACE/HANDS frames are available for further view-specific inspection. |

These observations invalidate several confident historical stories, while preserving original GT and frozen scoring. The ledger retains `UNVERIFIED_ROOT_CAUSE` where visual and trace evidence do not establish the first incorrect stage. The separate boundary experiment uses autonomous B0 observations only; these GT-selected images never enter its inference interface.
