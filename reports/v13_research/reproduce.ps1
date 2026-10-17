$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

function Invoke-NativeChecked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Native command failed: $LASTEXITCODE" }
}

Invoke-NativeChecked { py -m training.dmd.v13_diagnostics }
Invoke-NativeChecked { py -m compileall -q backend training tests 2>&1 | Tee-Object reports/v13_research/compile.log }
Invoke-NativeChecked { py -m ruff check training/dmd/behavior_roi.py training/dmd/high_res_roi.py training/dmd/v13_diagnostics.py training/dmd/v13_reports.py training/dmd/verify_v13.py tests/test_v12_research.py 2>&1 | Tee-Object reports/v13_research/ruff.log }
Invoke-NativeChecked { py -m pytest -q tests/test_v12_research.py tests/test_v11_research.py --junitxml=reports/v13_research/pytest_targeted.xml 2>&1 | Tee-Object reports/v13_research/pytest_targeted.log }
Invoke-NativeChecked { py -m pytest -q --junitxml=reports/v13_research/pytest_full.xml 2>&1 | Tee-Object reports/v13_research/pytest_full.log }
Push-Location frontend
try { Invoke-NativeChecked { npm run build 2>&1 | Tee-Object ..\reports\v13_research\frontend_build.log } }
finally { Pop-Location }
Invoke-NativeChecked { py -c "from backend.main import app; import json; print(json.dumps({'status':'PASS','paths':len(app.openapi()['paths'])}))" 2>&1 | Tee-Object reports/v13_research/backend_smoke.log }
Invoke-NativeChecked { py -m training.dmd.v13_reports }
Invoke-NativeChecked { py -m training.dmd.verify_v13 }
