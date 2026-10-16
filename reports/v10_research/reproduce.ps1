$ErrorActionPreference = "Stop"
$repoRoot = (Resolve-Path (Join-Path $PSScriptRoot "..\..")).Path
Set-Location -LiteralPath $repoRoot

py -3 -m training.dmd.v10_representation_probe
py -3 -m training.dmd.v10_root_cause
py -3 -m training.dmd.v10_audit
py -3 -m training.dmd.v10_synthesis
py -3 -m ruff check training/dmd/v10_audit.py training/dmd/v10_representation_probe.py training/dmd/v10_root_cause.py training/dmd/v10_synthesis.py training/dmd/verify_v10.py tests/test_v10_research.py
py -3 -m compileall -q backend training tests
py -3 -m pytest tests/test_v10_research.py tests/test_v9_research.py tests/test_holdout_guard.py tests/test_event_evaluation_governance.py tests/test_context_truth_governance.py tests/test_ai85_plus.py tests/test_ai85_next.py tests/test_review1_governance.py -q --junitxml=reports/v10_research/pytest_targeted.xml 2>&1 | Tee-Object -FilePath reports/v10_research/pytest_targeted.log
py -3 -m pytest -q --junitxml=reports/v10_research/pytest_full.xml 2>&1 | Tee-Object -FilePath reports/v10_research/pytest_full.log
Push-Location frontend
try {
    npm run build 2>&1 | Tee-Object -FilePath ..\reports\v10_research\frontend_build.log
}
finally {
    Pop-Location
}
py -3 -c "import json; from backend.main import app; schema=app.openapi(); print(json.dumps({'status':'PASS','title':schema['info']['title'],'version':schema['info']['version'],'paths':len(schema['paths'])}))" 2>&1 | Tee-Object -FilePath reports/v10_research/backend_smoke.log
py -3 -m training.dmd.v10_test_report
py -3 -m training.dmd.verify_v10
