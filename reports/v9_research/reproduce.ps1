$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "../..")
Push-Location $RepoRoot
try {
    py -3.11 -m training.dmd.v9_diagnostics
    if ($LASTEXITCODE -ne 0) { throw "V9 diagnostics failed" }
    $reviewReceipts = @(Get-ChildItem "reports/v9_research/human_phone_reviews" -File -Filter "*.json" -ErrorAction SilentlyContinue)
    if ($reviewReceipts.Count -eq 0) {
        py -3.11 -m training.dmd.v9_phone_proposal_mining
        if ($LASTEXITCODE -ne 0) { throw "V9 phone proposal mining failed" }
    } elseif ($reviewReceipts.Count -ne 13) {
        throw "V9 human review is partially complete; expected 0 or 13 latest receipts"
    }
    py -3.11 -m training.dmd.v9_human_review_results
    if ($LASTEXITCODE -ne 0) { throw "V9 human review verification failed" }
    py -3.11 -m training.dmd.v9_phone_context_ablation
    if ($LASTEXITCODE -ne 0) { throw "V9 phone-context ablation failed" }
    py -3.11 -m training.dmd.render_v9_reports
    if ($LASTEXITCODE -ne 0) { throw "V9 report rendering failed" }
    py -3.11 -m training.dmd.verify_v9 --checks
    if ($LASTEXITCODE -ne 0) { throw "V9 verification failed" }
}
finally {
    Pop-Location
}
