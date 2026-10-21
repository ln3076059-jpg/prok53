# ROADWATCH V25 — Final AI lead decision

Generated: 2026-10-21T00:56:33.310850+07:00

`DEVELOPMENT_TARGET_NOT_MET`. `INDEPENDENT_TARGET_NOT_ESTABLISHED`.
`production_approved=false`.

The V24 zero-forward defect is fixed and verified. E4 and E5 each execute 1,803
deep forwards (787 on gC-14 and 1,016 on gZ-36). Correct execution does not improve
the candidate beyond the frozen cached-replay baseline:

| Run | TP | FP | FN | Precision | Recall | F1 |
|---|---:|---:|---:|---:|---:|---:|
| E0 legacy-role baseline | 13 | 7 | 7 | 0.650000 | 0.650000 | 0.650000 |
| E2 association/lifecycle | 5 | 0 | 15 | 1.000000 | 0.250000 | 0.400000 |
| E3 rule-only | 6 | 1 | 14 | 0.857143 | 0.300000 | 0.444444 |
| E4 deep-only | 5 | 1 | 15 | 0.833333 | 0.250000 | 0.384615 |
| E5 hybrid | 6 | 0 | 14 | 1.000000 | 0.300000 | 0.461538 |

Best measured V25 result is E0 with F1=0.650000; it uses legacy
role assumptions and is not a verified-driver production candidate. The strongest
verified-geometry candidate is E5 at F1=0.461538. It has high
precision but recall=0.300000, leaving 14 of
20 events unmatched.

Detector and association >0.85 gates remain `NOT_ESTABLISHED` because the discovered
artifacts do not provide the exhaustive bbox or independent role ground truth required
by the metric contract. Seatbelt and historical action metrics remain below target.

Tests: 88 passed, 0 failed, 0 errors, 0 skipped. Full-video V25
candidate evaluation was not run; V24 E8 remains a 30-sample component runtime smoke.
Training remains blocked by the current data-governance record (`RIGHTS_ACCEPTED=false`).
