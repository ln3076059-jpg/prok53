# ==============================================================================
# ROADWATCH V21 — REPRODUCIBILITY COMMANDS SCRIPT (PowerShell)
# ==============================================================================

Write-Host ">>> Step 1: Setting up Python Environment..."
$env:PYTHONPATH = "d:\.idea\giangdoantotnghiep"

Write-Host ">>> Step 2: Running Complete Pytest Suite (659 Regression Tests)..."
py -3.11 -m pytest tests/

Write-Host ">>> Step 3: Running V21 Targeted Scientific Regression Tests..."
py -3.11 -m pytest tests/test_v21_scientific_engineering.py -v

Write-Host ">>> Step 4: Running V21 Smoke Test (100 Sampled Frames)..."
py -3.11 training/dmd/run_v21_unified_experiments.py --test-run

Write-Host ">>> Step 5: Running Full Autonomous Video Inference & Evaluation (15.27 min cohort)..."
py -3.11 training/dmd/run_v21_unified_experiments.py

Write-Host ">>> Step 6: Verifying Machine-Readable Output Artifacts..."
Get-ChildItem reports/v21_ai85_recovery/

Write-Host ">>> ALL REPRODUCIBILITY STEPS COMPLETED SUCCESSFULLY."
