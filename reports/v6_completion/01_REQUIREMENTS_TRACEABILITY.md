# V6 Requirements Traceability

Scope reviewed at commit `9c0e872277a02ae99f30dbecad2195333d23423a` on branch `roadwatch-v4-candidate`. V6 remains research-only.

| Requirement | Evidence | Status |
|---|---|---|
| Protect Git/worktree and consumed holdouts | `holdouts_accessed=[]`; no reset/clean; gZ-37 and gE-28 excluded by guard | PASS |
| Verify V4/V5 baselines | V5 round-2 JSON and Section 09 table | PASS |
| Diagnose class collapse | data audit plus four controlled V6 variants | PASS |
| Avoid outer-label threshold tuning | fixed 0.80 threshold declared before V6 result generation | PASS |
| Evaluate phone events per subject | LOSO fold artifacts and V6 event tables | PASS |
| Evaluate seatbelt model | 641 raw predictions, confusion, ECE and abstention recomputed | PASS WITH LIMITS |
| Keep research models out of production | registry state DEVELOPMENT_ONLY; runtime production gate unchanged | PASS |
| Upload → worker → evidence → review/export workflow | backend tests and repository audit | IMPLEMENTED; integration test status in Section 10 |
| Frontend workflow | login, upload, progress, events, evidence, review, statistics, filters/export present | IMPLEMENTED |
| Runtime configuration | `.env.example` now targets V2 rather than UNTRAINED config; `.dockerignore` added | PASS |
| Full verification | compile, Ruff, pytest, backend smoke, frontend build captured in JSON | SEE SECTION 10 |
| Real inference performance | local decode, YOLO, seatbelt, fusion, evidence, ResNet, head and backend pipeline measured | PASS |
| Roadside scientific claim | no roadside ground-truth benchmark exists | NOT ESTABLISHED |
| Production approval | no independent new holdout or human approval | BLOCKED BY DESIGN |

The thesis system is demonstrable for cabin-video scope. It must not be described as validated roadside enforcement.
