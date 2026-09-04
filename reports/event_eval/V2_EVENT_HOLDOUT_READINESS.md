# V2 Event Holdout Readiness

**Status:** `PENDING_NEW_UNTOUCHED_HOLDOUT`

## Overview
Because the first canonical frozen test has been consumed (`FROZEN_TEST_RUN_COUNT = 1`) to validate raw model capabilities without calibration, it cannot be reused. Reusing it to tune temporal policies or threshold bounds would constitute data leakage (overfitting on the test set).

## Requirements for Event Holdout
To proceed to End-to-End Event Evaluation, we require a new, completely untouched holdout dataset that meets these strict invariants:
1. **Never** used in model training.
2. **Never** used in validation or confidence sweep.
3. **Never** used in temporal or threshold calibration.
4. **Never** used in the original canonical frozen test.
5. No overlap in clips, subjects, or adjacent frames with any existing dataset.
6. Must contain sequence-level ground truth events (start/end frames of violation).

Once this dataset is prepared and fingerprinted, we can execute the end-to-end evaluation pipeline.
