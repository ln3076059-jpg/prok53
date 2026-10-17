$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

py -m training.dmd.v11_audit
py -m training.dmd.v11_fn_audit
py -m training.dmd.v11_representation_experiments
py -m training.dmd.v11_generalization_experiments
py -m training.dmd.v11_reports

New-Item -ItemType File reports/v11_research/compile.log -Force | Out-Null
py -m compileall -q backend training tests 2>&1 | Tee-Object reports/v11_research/compile.log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
New-Item -ItemType File reports/v11_research/ruff.log -Force | Out-Null
py -m ruff check training/dmd/behavior_roi.py training/dmd/causal_motion.py training/dmd/driver_phone_association.py training/dmd/v11_audit.py training/dmd/v11_fn_audit.py training/dmd/v11_representation_experiments.py training/dmd/v11_generalization_experiments.py training/dmd/v11_reports.py training/dmd/v11_test_report.py training/dmd/verify_v11.py tests/test_v11_research.py 2>&1 | Tee-Object reports/v11_research/ruff.log
if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
py -m pytest -q tests/test_v11_research.py --junitxml=reports/v11_research/pytest_targeted.xml 2>&1 | Tee-Object reports/v11_research/pytest_targeted.log
py -m pytest -q --junitxml=reports/v11_research/pytest_full.xml 2>&1 | Tee-Object reports/v11_research/pytest_full.log

Push-Location frontend
npm run build 2>&1 | Tee-Object ..\reports\v11_research\frontend_build.log
Pop-Location

py -c "from backend.main import app; import json; print(json.dumps({'status':'PASS','paths':len(app.openapi()['paths'])}))" 2>&1 | Tee-Object reports/v11_research/backend_smoke.log
py -m training.dmd.v11_test_report
py -m training.dmd.verify_v11
