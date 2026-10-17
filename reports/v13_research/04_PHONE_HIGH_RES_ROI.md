# V13 phone high-resolution ROI

V13 decoded 13 original BODY frames from hash-verified sources and evaluated R0-R5 extraction. There are 12 authenticated reviewed boxes; R5 preserved full phone coverage for 12/12. The remaining case has no valid phone proposal and stays a hard example. Original frame coordinates, content hashes, output sizes, transform provenance, edge clipping and coverage are recorded in `high_resolution_roi.json`.

R0 remains the frozen 64x64 reference. R1 preserves the original driver crop, R2 uses padded phone/hand evidence, R3 records a face/hand fallback when those boxes are absent, R4 returns multiple source-resolution regions, and R5 uses a scale-aware dynamic crop. These are oracle-box diagnostics. Detector confidence was not regenerated, candidate yield did not change, Phone AP is NOT MEASURED, and zero events are claimed recovered.
