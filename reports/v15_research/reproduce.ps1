$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
function Run([scriptblock]$Command) { & $Command; if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" } }
Run { py -3.11 -m training.dmd.v15_diagnostics }
Run { py -3.11 -m compileall -q backend training tests 2>&1 | Tee-Object reports/v15_research/compile.log; "PASS" | Add-Content reports/v15_research/compile.log }
Run { py -3.11 -m ruff check training/dmd/autonomous_phone.py training/dmd/sliced_phone.py training/dmd/v15_diagnostics.py training/dmd/v15_reports.py training/dmd/verify_v15.py tests/test_v15_sliced_phone.py 2>&1 | Tee-Object reports/v15_research/ruff.log }
Run { py -3.11 -m pytest -q tests/test_v15_sliced_phone.py tests/test_v12_research.py --junitxml=reports/v15_research/pytest_targeted.xml 2>&1 | Tee-Object reports/v15_research/pytest_targeted.log }
Run { py -3.11 -m pytest -q --junitxml=reports/v15_research/pytest_full.xml 2>&1 | Tee-Object reports/v15_research/pytest_full.log }
Push-Location frontend
try { Run { npm run build 2>&1 | Tee-Object ..\reports\v15_research\frontend_build.log } } finally { Pop-Location }
Run { py -3.11 -c "from backend.main import app; import json; print(json.dumps({'status':'PASS','paths':len(app.openapi()['paths'])}))" 2>&1 | Tee-Object reports/v15_research/backend_smoke.log }
Run { py -3.11 -m training.dmd.v15_reports }
Run { py -3.11 -m training.dmd.verify_v15 }
