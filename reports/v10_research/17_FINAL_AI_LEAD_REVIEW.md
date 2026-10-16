# V10 final AI lead review

Decision: **REJECT FOR PRODUCTION; CONTINUE CONTROLLED RESEARCH**.

The new measured result is a LEVEL 2 representation probe. Temporal residual D2 improves pooled clip
Macro-F1 to 51.54%, but worst-subject SAFE recall is 1.66%, action silhouette
falls, the representation is non-causal, and no event evaluation exists. The FN audit shows 12/13 reviewed
cases have usable auxiliary phone boxes, yet association is unverified in all 13 and dense action evidence
crosses the gate in only 4. These findings support representation, association, and temporal work; they do not
support deployment.

The controlled event comparator remains P=80.00%, R=55.17%,
F1=65.31%, FA/min=0.1813. Seatbelt best development
Macro-F1 is 58.52%. No independent holdout exists, rights are
unaccepted, and protected subjects were not opened. `production_approved=false`; no runtime config or active
checkpoint was changed.

Next admissible work is: obtain explicit data rights and new approved subjects; lock a group-disjoint holdout;
extract higher-resolution behavior/hand/face regions; build occupant association with UNKNOWN handling; turn D2
into a causal motion encoder; then evaluate event IDs, onset latency, short events, fragmentation, and all
negative contexts under a frozen policy. Seatbelt training waits for human-approved three-state labels.

Artifact inventory: 64 local checkpoints hashed without deserialization. Source
hashes and machine-readable metrics are in `model_manifest.json`, `experiment_registry.json`, and
`metrics_comparison.json`.
