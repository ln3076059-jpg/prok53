$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
function Run([scriptblock]$Command) { & $Command; if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" } }
Run { py -m training.dmd.v14_diagnostics }
Run { py -m compileall -q backend training tests 2>&1 | Tee-Object reports/v14_research/compile.log }
Run { py -m ruff check training/dmd/autonomous_phone.py training/dmd/behavior_roi.py training/dmd/high_res_roi.py training/dmd/online_candidate.py training/dmd/v14_diagnostics.py training/dmd/v14_reports.py training/dmd/verify_v14.py tests/test_v12_research.py 2>&1 | Tee-Object reports/v14_research/ruff.log }
Run { py -m pytest -q tests/test_v12_research.py tests/test_v11_research.py --junitxml=reports/v14_research/pytest_targeted.xml 2>&1 | Tee-Object reports/v14_research/pytest_targeted.log }
Run { py -m pytest -q --junitxml=reports/v14_research/pytest_full.xml 2>&1 | Tee-Object reports/v14_research/pytest_full.log }
Push-Location frontend
try { Run { npm run build 2>&1 | Tee-Object ..\reports\v14_research\frontend_build.log } } finally { Pop-Location }
Run { py -c "from backend.main import app; import json; print(json.dumps({'status':'PASS','paths':len(app.openapi()['paths'])}))" 2>&1 | Tee-Object reports/v14_research/backend_smoke.log }
Run { py -m training.dmd.v14_reports }
Run { py -m training.dmd.verify_v14 }
