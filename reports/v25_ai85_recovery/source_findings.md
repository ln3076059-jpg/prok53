# V25 source and evidence findings

Generated: 2026-10-21T00:56:33.309363+07:00

1. `replay_v24_record()` previously called `observe_embedding()` and then
   `predict_from_embedding()` at the same timestamp. The second call reset the
   temporal buffer. V24 E4/E5/E6 therefore loaded checkpoints but recorded zero
   model forwards.
2. The adapter now separates state mutation from scoring. `predict_from_sequence()`
   scores an existing 9x512 window and records the center target time, availability
   time and emission latency. Duplicate feature delivery is idempotent; out-of-order
   delivery fails closed.
3. Cache lookup now binds `frame_idx` to feature index and tolerates only the
   sub-millisecond timestamp rounding of the same frame. It does not substitute a
   different future feature.
4. Driver identity is stable by calibrated seat, and pose evidence is selected by
   the recorded source pose index. Multiple driver poses in one seat are ambiguous
   and fail closed.
5. V25 E4 executed 1803 forwards; E5 executed 1803.
   Correct execution did not meet the quality target.
6. The original 20-event ground truth and frozen matcher were preserved. No
   annotation normalization, fitting, upload, holdout access or production activation
   was performed.
