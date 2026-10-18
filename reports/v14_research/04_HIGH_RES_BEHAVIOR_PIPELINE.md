# V14 high-resolution behavior pipeline

Autonomous proposals now feed the V13 `HighResolutionRoiPipeline` directly in original-frame coordinates. No 64x64 cache is enlarged and described as recovered detail. Crop records include deterministic frame SHA-256, input/output dimensions, requested and clipped boxes, padding, scale, evidence state, driver track ID and phone coverage when a runtime proposal exists.

Synthetic regression cases verify partial boundary coverage and coordinate local-to-original round trips. When a proposal is absent, the pipeline uses available hand/face evidence or an upper-body fallback with UNKNOWN evidence semantics. The V13 oracle result of 12/12 reviewed-box coverage remains explicitly separate as ORACLE_BOX_DIAGNOSTIC_ONLY and is not an autonomous benchmark result.
