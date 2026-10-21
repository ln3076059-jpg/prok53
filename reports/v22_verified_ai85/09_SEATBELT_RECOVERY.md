# Seatbelt recovery audit

Recomputed existing 641-row validation CSV with sklearn reconciliation. Raw Macro-F1=0.572753; post-policy Macro-F1=0.555982; UNCERTAIN recall=0.040000. This is cached validation prediction evidence, not new model inference or independent performance.

Different historical 58.52% result uses a different policy/TTA variant; do not choose the higher score as a new baseline. Current CSV lacks subject IDs and exact checkpoint linkage, so worst-subject metrics and full provenance remain blocked. seatbelt_metrics_recomputed.json includes per-class F1, false-unfastened rate and abstention.

No pending labels were approved, no rights flags changed and no warm-start training performed. Need human-reviewed difficult uncertain/occluded torso cases, identity/session groups, clear rights and a locked subject split before fitting.
