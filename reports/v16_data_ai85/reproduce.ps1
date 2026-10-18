$ErrorActionPreference = "Stop"
$root = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location $root
function Run([scriptblock]$Command) { & $Command; if ($LASTEXITCODE -ne 0) { throw "exit $LASTEXITCODE" } }
Run { py -3.11 -m training.dmd.v16_data_audit }
Run { py -3.11 -m compileall -q backend training tests 2>&1 | Tee-Object reports/v16_data_ai85/compile.log; "PASS" | Add-Content reports/v16_data_ai85/compile.log }
Run { py -3.11 -m ruff check training/data_governance.py training/download_source.py training/ingest_dataset.py training/split_dataset.py training/dmd/v16_data_audit.py training/dmd/v16_reports.py training/dmd/verify_v16.py tests/test_v16_data_governance.py 2>&1 | Tee-Object reports/v16_data_ai85/ruff.log }
Run { py -3.11 -m pytest -q tests/test_v16_data_governance.py tests/test_v15_sliced_phone.py tests/test_review1_governance.py --junitxml=reports/v16_data_ai85/pytest_targeted.xml 2>&1 | Tee-Object reports/v16_data_ai85/pytest_targeted.log }
Run { py -3.11 -m pytest -q --junitxml=reports/v16_data_ai85/pytest_full.xml 2>&1 | Tee-Object reports/v16_data_ai85/pytest_full.log }
Push-Location frontend
try { Run { npm run build 2>&1 | Tee-Object ..\reports\v16_data_ai85\frontend_build.log } } finally { Pop-Location }
Run { py -3.11 -c "from backend.main import app; import json; print(json.dumps({'status':'PASS','paths':len(app.openapi()['paths'])}))" 2>&1 | Tee-Object reports/v16_data_ai85/backend_smoke.log }
Run { py -3.11 -m training.dmd.v16_reports }
Run { py -3.11 -m training.dmd.verify_v16 2>&1 | Tee-Object reports/v16_data_ai85/verification.log }
