# Run from repository root, Python 3.11 with the installed pinned dependencies.
# Original models, allowed development media and cached baseline features are required.
# A new output folder protects existing experimental measurements.
param([string]$RunDirectory = "reports/ai85_plus_reproduction")
$ErrorActionPreference = "Stop"
if (Test-Path -LiteralPath $RunDirectory) {
    throw "Choose a fresh RunDirectory to preserve historical results."
}
py -3.11 -m training.dmd.audit_ai85_plus --output $RunDirectory
if ($LASTEXITCODE -ne 0) { throw "Artifact audit failed" }
py -3.11 -m training.dmd.ai85_plus_visual inventory --output $RunDirectory
if ($LASTEXITCODE -ne 0) { throw "Checkpoint inventory failed" }
py -3.11 -m training.dmd.ai85_plus_visual phone --output $RunDirectory
if ($LASTEXITCODE -ne 0) { throw "Phone probes failed" }
py -3.11 -m training.dmd.run_ai85_plus_temporal --output "$RunDirectory/temporal"
if ($LASTEXITCODE -ne 0) { throw "Temporal experiment failed" }
py -3.11 -m training.dmd.ai85_plus_diagnostics --output $RunDirectory
if ($LASTEXITCODE -ne 0) { throw "Evidence diagnostics failed" }
py -3.11 -m training.dmd.ai85_plus_diagnostics --migration-differences-only --output $RunDirectory
if ($LASTEXITCODE -ne 0) { throw "Migration difference audit failed" }
py -3.11 -m training.dmd.ai85_plus_visual seatbelt --output $RunDirectory
if ($LASTEXITCODE -ne 0) { throw "Seatbelt experiment failed" }
py -3.11 -m training.benchmark_v6_runtime --frames 24 --output "$RunDirectory/runtime_benchmark.json"
if ($LASTEXITCODE -ne 0) { throw "Runtime benchmark failed" }
py -3.11 -m training.dmd.ai85_plus_system_smoke --output "$RunDirectory/backend_smoke.json"
if ($LASTEXITCODE -ne 0) { throw "Backend smoke failed" }
py -3.11 -m training.dmd.verify_ai85_plus --checks --output $RunDirectory
if ($LASTEXITCODE -ne 0) { throw "Tests failed" }
py -3.11 -m training.dmd.render_ai85_plus --output $RunDirectory
if ($LASTEXITCODE -ne 0) { throw "Report rendering failed" }
py -3.11 -m training.dmd.verify_ai85_plus --output $RunDirectory
if ($LASTEXITCODE -ne 0) { throw "Artifact verification failed" }
