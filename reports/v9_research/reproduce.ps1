$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
Push-Location $RepoRoot
try {
    py -3.11 -m training.dmd.v9_diagnostics
    if ($LASTEXITCODE -ne 0) { throw "V9 diagnostics failed" }
    py -3.11 -m training.dmd.render_v9_reports
    if ($LASTEXITCODE -ne 0) { throw "V9 report rendering failed" }
    py -3.11 -m training.dmd.v9_phone_proposal_mining
    if ($LASTEXITCODE -ne 0) { throw "V9 phone proposal mining failed" }
    py -3.11 -m training.dmd.verify_v9 --checks
    if ($LASTEXITCODE -ne 0) { throw "V9 verification failed" }
}
finally {
    Pop-Location
}
