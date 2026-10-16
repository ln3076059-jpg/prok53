# Roadwatch V9 final scientific report

V9 completed the prerequisite investigation and reproducible instrumentation; it did not fine-tune or
approve a model. Only 2/13 FN events have any sampled phone box,
and none has verified occupant identity. Sampling density is downstream of this dominant failure.


Human review is now complete for 13/13 FN cases: 12 selected boxes and
1 visible-phone case without a valid box. The resulting lane is
hash-registered and excluded from automatic training. This resolves the review prerequisite but does not
provide an independent subject for model selection or evaluation.


Frozen embeddings separate subject identity 8.42×
more strongly than action. Dense sampling recovered one long event but raised false alarms. Seatbelt
development Macro-F1 peaks at 58.52%; 500 candidates remain pending.

- Event P/R/F1 ≥85%: **not achieved**.
- FA/min ≤0.15: **not established by an eligible candidate**.
- SAFE_DRIVING recall ≥85%: **not achieved**.
- Seatbelt Macro-F1 ≥85%: **not achieved**.
- Independent generalization: **not established**.
- Production approval: **false**.

The next model-changing run requires new group-disjoint subjects plus a frozen representation protocol.
The reviewed three-subject error lane can guide ROI and small-object hypotheses but cannot be used to
claim recovery on those same cases. Keep `gZ-37` and `gE-28` sealed until model, threshold and event policy
are frozen.


The strict geometry gate is now implemented with raw/admitted confidence and rejection-reason audit
fields. Its diagnostic removes 99 mounted/static candidates while retaining all
12 selected boxes at the mining floor. It remains opt-in until callers supply
audited context; no event metric changed in this run.
