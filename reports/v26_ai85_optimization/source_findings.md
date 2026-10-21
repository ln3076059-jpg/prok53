# ROADWATCH V26 source findings

Generated: 2026-10-21T06:39:18.496177+07:00

1. The cached active-boundary reporter did not call `TemporalEventEngine.expire()`, while
   `VideoAnalyzer` calls it on every frame. A verified driver event therefore survived an
   unknown-pose gap and was extended when the driver reappeared. On gZ-36 this produced an
   interval from 52.9 to 376.5 seconds.
2. The reporter now expires stale windows before every observation and records lifecycle
   transitions in raw traces. A regression test proves that two positive episodes separated
   by an unknown-pose gap remain two events.
3. Under the unchanged 20-event matcher, E5 recall increased from 0.30 to 0.60. Precision
   fell because previously hidden bursts became separate predictions. E2 is the best corrected
   run at F1=0.533333.
4. The remaining FN evidence includes sub-second intervals below temporal support, intervals
   with no exported phone candidate, and intervals with sparse/missing verified association.
   The cached trace does not contain pre-filter detector proposals, so detector AP/recall is
   not established.
5. Action raw records now expose rule PHONECALL/TEXTING scores and predicted subtype for
   diagnostics. No subtype boundary policy was fitted or introduced because explicit training,
   label and group-split approvals are absent from the live governance record.
6. The first full-video run detected 262 phone
   candidates but marked all 262 associations
   UNKNOWN because driver-seat poses were always constructed as unverified. The shared raw
   runner now maps valid poses through the recorded BODY-camera seat ROI and
   fails closed when pose geometry is missing or invalid. The manifest's VERIFIED label
   is a project assumption, not independent role ground truth.
7. Full raw-video D2 status after that fix is PARTIAL. Completed subjects:
   gZ-36. Aggregate event F1 is
   0.421053; this is repeatedly inspected development evidence.
8. Cached E2-E5 consume historical D5 proposals (specialist plus auxiliary), whereas
   these raw runs use D2 (specialist only). They are not a detector-controlled parity test.
9. See `ROOT_CAUSE_AND_LITERATURE_REVIEW.md` for corrected event attribution, source
   comparisons and limitations. These raw metrics precede the latest boundary collector;
   they must not be attributed to unexecuted current code.
