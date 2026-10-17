# V13 final AI lead report

Decision: **RESEARCH_ONLY; REJECT FOR PRODUCTION**.

V13 changed `behavior_roi.py`, added `high_res_roi.py`, V13 diagnostics/report tooling and regression coverage. It ran live permission checks, 13 original-frame R0-R5 diagnostics, baseline/artifact reconciliation, targeted and full tests, frontend build and backend smoke. The measurable new result is engineering evidence: 12/12 reviewed phone boxes retain full coverage in the source-resolution R5 crop, at ROI-suite p50 25.939 ms. This is not an accuracy result.

Phone events remain 16/4/13 with P/R/F1 80.00%/55.17%/65.31% and FA/min 0.1813. E3 clip Macro-F1 remains 56.39%, SAFE/PHONE/OTHER recall 60.38%/66.78%/43.26%, worst-subject SAFE recall 54.36%. Seatbelt best development Macro-F1 remains 58.52%, UNCERTAIN recall 4%. Phone AP and E3 Event F1 remain NOT MEASURED.

There is no fitted E3 head/scaler, verified occupant ground truth, approved seatbelt label set, valid fitting authorization, new group-disjoint validation or independent holdout. The 85% objective is not scientifically established. The highest-value next experiment is a rights-approved, subject-disjoint fit comparing frozen R0 against R5 plus causal E3, with validation-fixed calibration and event policy before a locked independent test. Production approval remains false.
