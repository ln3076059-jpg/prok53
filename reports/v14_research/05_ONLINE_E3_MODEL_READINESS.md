# V14 OnlineE3 model readiness

OnlineE3 causal features remain engineering-ready with inherited batch/stream parity. NumericHead now accepts strict schema v2 with dataset/split/feature provenance plus checkpoint hash, config hash, library versions, exact class order and an explicit calibration state. Loading can verify the SHA-256 of the complete JSON file before parsing. Decision-function argmax remains authoritative; normalized OvR sigmoid scores remain uncalibrated unless a locked-validation calibration record says otherwise.

The real fitted head/scaler status remains `BLOCKED_MISSING`. Schema v2 was tested with synthetic numeric fixtures only. No synthetic coefficient is reported as a model and E3 Event F1 remains NOT MEASURED.
