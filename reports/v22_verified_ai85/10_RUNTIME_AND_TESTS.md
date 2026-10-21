# Runtime and validation

runtime_benchmark.json contains actual CPU session extraction timings and separate cached-head/replay timings. Torch is CPU-only; no GPU speed is reported. Shared detector extraction is reused by B0/B1/B2, not multiplied into independent runtime claims.

tests_summary.json is parsed from current JUnit XML and recorded native exit codes. logs preserve failing-before and passing-after evidence. No inherited 667-pass count is treated as current verification.

Engineering remains partial: unsafe fallbacks, coverage loss and role fabrication paths are fixed, but full deep/action/association scientific validation is blocked. No production model was activated.
