# ROADWATCH V19 — Master Reproduction Commands (Windows PowerShell)

Set-Location "D:\.idea\giangdoantotnghiep"

# 1. Verify Git status and HEAD
git status --short
git branch --show-current
git rev-parse HEAD

# 2. Verify V18 and V19 test suites
py -m pytest tests/test_v18_bug_fixes.py tests/test_v19_metric_recovery.py -v

# 3. Execute Controlled Event Replay R0-R5
py -m training.dmd.v19_bug_fix_replay

# 4. Generate all V19 recovery reports and JSON artifacts
py -m training.dmd.generate_v19_research_reports

# 5. Lint and syntax compilation check
py -m ruff check training/dmd/v19_bug_fix_replay.py tests/test_v19_metric_recovery.py
py -m compileall -q backend training tests
