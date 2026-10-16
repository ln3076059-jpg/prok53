# GitHub artifact scope

This directory publishes the AI85 experiment reports, metrics, fold predictions,
configuration and source snapshots, extraction metadata, hashes, and validation
logs. The experiments did not achieve the 85% target and no candidate was
activated in production; see `15_FINAL_REPORT.md` for the measured results.

Original dataset ROI images in `fn_original_rois/`, extracted feature caches in
`native112/embeddings/*.npz`, and model checkpoint weights (`*.pt`) remain local
and are excluded from Git. Reproducing the experiments requires the corresponding
datasets and checkpoint weights. Artifact inventories and validation records
describe the local experiment run, including these unpublished artifacts; they
are not a list of files available from a GitHub checkout.

The recorded targeted suite passed 89 tests. The full suite recorded 425 passed,
37 failed, 74 errors, and 1 skipped; the failures and errors are classified in
`test_failure_classification.json` and concern missing pre-existing dataset policy
and schema files. See `14_TEST_RESULTS.md` for the validation limits.
