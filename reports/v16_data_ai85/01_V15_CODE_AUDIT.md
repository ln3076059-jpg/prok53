# V15 code and evidence audit

V16 started from local commit `25d90053a0a41fb483aaa43038d8e6c23ebbb12a` and preserved the user-owned
`prompt.md` change. The V15 implementation is present in `training/dmd/sliced_phone.py` and is
wired into `AutonomousPhonePipeline.detect_roi`. It contains overlapping tile generation,
coordinate projection back to the source ROI, edge penalties, containment handling and NMS.
No second SAHI adapter was created.

The V15 detector checkpoints remain unchanged. The specialist checkpoint SHA-256 is
`840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54`; the auxiliary
checkpoint SHA-256 is `0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1`.
V15 reported 40 targeted and 629 full tests. V16 reruns are recorded in
`test_results.json`, not inferred from that historical claim.

The accuracy limitation remains decisive: V15 has engineering tests and a diagnostic proposal
yield result, but no approved positive-box plus reviewed-negative detector validation set.
Therefore V15 did not establish Phone AP or an event-metric improvement.
