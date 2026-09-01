# V2 Model Lock Protocol

Status: **UNLOCKED**. A `best.pt` file is not a model lock.

The candidate lock must bind:

- experiment and code commit;
- phone, upper-body, classifier and configured windshield weight SHA-256 values;
- runtime config SHA-256;
- governed training-manifest and split hashes;
- validation metrics and threshold-calibration artifact hashes;
- fusion schema/artifact hash when calibrated fusion is required;
- the human-review readiness artifact and its governed-ready state;
- creation time, activation state and explicit production approval state.

Lock creation fails on a missing artifact, hash mismatch, placeholder threshold, non-governed
readiness or missing required fusion. Runtime production startup independently recomputes the
config/component hashes and rejects a non-ACTIVE or non-human-approved record.

`reports/model_lock_v2.json` does not currently exist because weights, governed data, calibration
and human approval are absent. Do not create a placeholder lock to make the path exist.
