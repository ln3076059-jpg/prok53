# Test results

Generated test evidence is stored in `test_results.json` and the referenced JUnit XML files.

- Targeted AI85-next tests: `PASS: 7 passed`.
- AI85 combined regression tests: `PASS: 16 passed`.
- Ruff: `PASS`.
- Compile: `PASS`.
- Full repository suite: `FAIL: 432 passed, 37 failed, 74 errors, 1 skipped`.

No failing test is removed or skipped to improve the report. Full-suite failures, if present, are
classified in `test_results.json`.
