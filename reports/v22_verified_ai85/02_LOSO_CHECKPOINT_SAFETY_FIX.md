# LOSO checkpoint safety

Removed both V6 and V7 other-fold fallbacks. Exact subject path is deterministic. Require matching SHA, split, architecture, preprocessing, class mapping, dataset provenance and backbone identity before online activation. ERROR mode fails closed; explicit RULE_ONLY mode records requested/actual mode, missing manifest reason and model_loaded=false.

Regression failing_before.log demonstrates deserialization of the wrong fold before the fix. Targeted/full test logs contain current outcomes. A matching checkpoint filename is insufficient proof.

Six original heads were loaded strictly and executed offline against development feature caches. checkpoint_provenance.json records real checkpoint hashes and split metadata. These diagnostics do not create an online compatibility manifest or retroactively approve data rights.
