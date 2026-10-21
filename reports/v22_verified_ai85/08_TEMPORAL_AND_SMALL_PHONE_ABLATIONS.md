# Temporal boundaries and detector feasibility

Frozen matcher bytes and policy bytes are unchanged. Matcher can attach multiple predictions to one GT; scoring counts duplicates as FP. This is not a strict one-to-one implementation, although unique-GT TP accounting is preserved. offset_tolerance_seconds is not used by matching. Published V21 0.05/4s policy does not match actual frozen .3/2.5s policy.

full_event_matches.json reports IoU, onset/offset errors, actual prediction/GT durations, union coverage and fragments separately. Temporal engine emits activation windows once; it does not finalize complete event boundaries on release or EOF. No synthetic EOF tail is added. A short alert is not a fully tracked GT event.

Official baseline uses stride15. Stride10/8/adaptive and SAHI AP ablations have not been run; no fully reviewed exhaustive original-frame detection annotation set was established. Additional phone proposals alone are not AP/recall evidence. Multiview fusion also remains blocked by sync/role evidence. These are explicit unperformed experiments, not measured improvements.

Separate T1 boundary replay: TP=13, FP=7, FN=7; Precision=0.650000, Recall=0.650000, F1=0.650000, FA/min=0.458400 on 15.270497 minutes / 20 GT. Reuses exact hashed B0 autonomous observations and the unchanged event engine/matcher; only interval reporting changes. Ends advance only to physically positive observations while the engine is active. EOF does not invent a tail. This is exploratory development evidence with legacy B0 driver-role assumptions; it does not establish verified association or independent AI85. Protocol and summary are boundary_experiment_protocol.json / boundary_experiment_summary.json; detailed matches stay under private/.

Retention check: recovered 8 original FN; 6 original FN remain; lost 1 original TP. FP reduction is under the frozen event accounting, not proof of rejecting nine physically false phone boxes. Changed boundaries can alter greedy event assignment; inspect the recovered/lost event IDs before adoption.
