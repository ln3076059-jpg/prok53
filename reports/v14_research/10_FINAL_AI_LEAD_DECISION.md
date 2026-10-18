# V14 final AI lead decision

Decision: **RESEARCH_ONLY; REJECT FOR PRODUCTION**.

KEEP engineering: autonomous frame-to-proposal adapter, source-resolution ROI integration, coordinate/provenance tests, missing-proposal UNKNOWN handling and NumericHead schema v2/hash validation. INCONCLUSIVE accuracy: autonomous detector proposal yield, because bbox negatives and a complete ground-truth denominator are absent. BLOCKED: action fitting, verified driver association, real event replay, seatbelt fitting, end-to-end benchmark and independent evaluation.

No metric reached 85% scientifically. Phone events remain P/R/F1 80.00%/55.17%/65.31%, FA/min 0.1813. E3 Macro-F1 remains 56.39%, SAFE/PHONE/OTHER recall 60.38%/66.78%/43.26%. Seatbelt Macro-F1 remains 58.52%, UNCERTAIN recall 4%. The highest-value next experiment is a rights-approved, group-disjoint dataset with autonomous bbox ground truth and verified occupant roles, followed by a frozen R5+E3 fit and fixed-policy replay. Production approval remains false.
