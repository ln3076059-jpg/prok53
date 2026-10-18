# Final AI lead review

Decision: **RESEARCH ONLY — REJECT FOR PRODUCTION**.

V15 verified the literature, audited runtime/research data flow, reconciled every one of the 13 false negatives, implemented a fixed-checkpoint-compatible tiled phone adapter, and strengthened geometry/abstention regression coverage. The dominant observed failures overlap: 11 specialist misses, 13 unknown driver associations and 9 action-gate failures. The adapter is scientifically relevant but has no valid accuracy benchmark yet.

No accuracy metric changed. Phone Event P/R/F1 remains 80.00%/55.17%/65.31% with 0.1813 FA/min. E3 Macro-F1 remains 56.39%, SAFE recall 60.38%, PHONE recall 66.78%, OTHER recall 43.26%, worst-subject SAFE recall 54.36%. Seatbelt Macro-F1 remains 58.52% and UNCERTAIN recall 4%. Independent AI85 is not established; `production_approved=false`.

Blockers are rights denial, absence of a complete positive/negative phone bbox benchmark, missing verified driver-role GT, missing fitted E3 head/scaler, 500 pending seatbelt labels and no locked independent holdout. The single highest-value next experiment is an authorized group-disjoint detection set containing positive and negative BODY frames with phone boxes and driver-role labels, then a fixed-checkpoint A0/A2/A3 comparison including AP, negative FP and model-inclusive latency.
