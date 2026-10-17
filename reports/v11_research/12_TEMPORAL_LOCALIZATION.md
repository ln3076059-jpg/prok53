# V11 temporal localization

ActionFormer supports multi-scale temporal features and direct boundary regression, but a full reproduction was not admitted: there is no approved large event-training set, no CUDA GPU, and current rights do not permit external training. V11 only establishes causal clip features as input for a future lightweight boundary model.

No H1-H5 event prediction artifact was generated. Therefore onset error, offset error, fragmentation, duplicate alerts, duration-bin recall, and online alarm latency are NOT MEASURED for the V11 representation. The frozen event comparator and its matcher remain unchanged. Prior T1-T5 temporal experiments did not improve Event F1, and dense sampling alone missed four subsecond events.

A valid next experiment should freeze causal inputs, learn separate start/end evidence on approved training subjects, rank candidates without ground-truth extension, and emit complete event IDs. Evaluation must retain the existing matcher and report events below one second, 1-3 seconds, 3-10 seconds, over 10 seconds, and full-pipeline runtime. Until that artifact exists, no clip score is described as temporal event recovery.
