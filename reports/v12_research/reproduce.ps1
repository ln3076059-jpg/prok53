$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $repoRoot

function Invoke-NativeChecked {
    param([scriptblock]$Command)
    & $Command
    if ($LASTEXITCODE -ne 0) { throw "Native command failed with exit code $LASTEXITCODE" }
}

if (-not (Test-Path reports/v12_research/experiment_plan.json)) {
    throw "experiment_plan.json must be recorded before diagnostic execution"
}
Invoke-NativeChecked { py -m training.dmd.v12_diagnostics }
New-Item -ItemType File reports/v12_research/compile.log -Force | Out-Null
Invoke-NativeChecked { py -m compileall -q backend training tests 2>&1 | Tee-Object reports/v12_research/compile.log }
New-Item -ItemType File reports/v12_research/ruff.log -Force | Out-Null
Invoke-NativeChecked { py -m ruff check training/dmd/research_permissions.py training/dmd/online_candidate.py training/dmd/v12_diagnostics.py training/dmd/v12_reports.py training/dmd/verify_v12.py training/dmd/driver_phone_association.py training/dmd/v11_representation_experiments.py training/dmd/v11_generalization_experiments.py tests/test_v12_research.py tests/test_v11_research.py 2>&1 | Tee-Object reports/v12_research/ruff.log }
Invoke-NativeChecked { py -m pytest -q tests/test_v12_research.py tests/test_v11_research.py --junitxml=reports/v12_research/pytest_targeted.xml 2>&1 | Tee-Object reports/v12_research/pytest_targeted.log }
Invoke-NativeChecked { py -m pytest -q --junitxml=reports/v12_research/pytest_full.xml 2>&1 | Tee-Object reports/v12_research/pytest_full.log }
Push-Location frontend
try { Invoke-NativeChecked { npm run build 2>&1 | Tee-Object ..\reports\v12_research\frontend_build.log } }
finally { Pop-Location }
Invoke-NativeChecked { py -c "from backend.main import app; import json; print(json.dumps({'status':'PASS','paths':len(app.openapi()['paths'])}))" 2>&1 | Tee-Object reports/v12_research/backend_smoke.log }
Invoke-NativeChecked { py -m training.dmd.v12_reports }
Invoke-NativeChecked { py -m training.dmd.verify_v12 }
