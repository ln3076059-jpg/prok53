# Association and fusion study

Runtime association combines calibrated ROIs or cabin geometry with explicit confidence and returns `unknown` below threshold. The local tracker provides continuity but cannot infer driver role. All 13 FN cases lack verified occupant tracks, so driver-association accuracy is not measurable.

Runtime `LogisticFusion` accepts detector, vehicle, occupant, phone-context, pose, seatbelt and temporal features. When calibrated fusion is not required, it has a conservative rule fallback. Research multiview fusion is a separate path and must not silently replace runtime semantics. Missing FACE/HANDS remain unavailable, not negative evidence.

V15 retains raw candidates while unknown driver/handheld evidence remains UNKNOWN/NEEDS_REVIEW. Mounted phone/camera evidence is non-use. Decision: KEEP the abstention contract; BLOCK association/fusion accuracy claims until verified role GT and a fitted/calibrated artifact exist.
