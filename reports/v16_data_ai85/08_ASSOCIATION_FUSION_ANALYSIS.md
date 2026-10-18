# Association and fusion analysis

The existing driver-phone association, autonomous phone, online candidate, fusion and event
modules were retained. A new candidate-flow accuracy run would require verified occupant role,
phone-object truth and temporal event truth on the same authorized samples. Those conditions
are absent, so counts at Detected → Candidate → Driver-associated → Handheld-evidence →
Action-supported → Fusion-admitted → Temporal-activated → Event-matched were not fabricated.

The V15 context replay remains historical development evidence: it removed one FP without
recovering a TP on a two-subject slice. It is not an independent validation result. Ambiguous
driver/passenger or mounted-device evidence must enter a review state rather than be counted as
a confirmed TP.
