# V21 claim versus evidence

Audited local and remote HEAD: `98502a4ab740d3ede3df18a6e217fb1861c009e9`. Newer than prompt's 494ec429. User prompt retained. Source history proves adapter and runner were introduced in 494ec429; auxiliary change in that commit only normalizes Path input.

Historical V20 scope: HISTORICAL_FULL_COHORT_BASELINE, TP=6 FP=16 FN=14 on 15.27 min / 20 GT. This is inherited evidence, not a fresh run.

Current V21 artifacts were updated by 98502a4 after the prompt baseline. They now report full-cohort exposure, so the prompt's earlier one-event description is historical. Recomputed file coverage and current fresh B0 are evaluated separately; V21 metrics remain REPORTED_NOT_YET_FULLY_REPRODUCED.

See claim_audit.json for current table, counts/hashes and score constancy. No original 3600 hybrid truth-label records were located. ROI percentages are literals in build_v21_metadata.py, not computed validation metrics.

Source graph: pipeline_graph.json. Production VideoAnalyzer does not run this action adapter. A research benchmark does not establish production behavior.
