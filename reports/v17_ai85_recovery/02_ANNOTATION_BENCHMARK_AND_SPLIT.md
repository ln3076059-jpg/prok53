# Annotation benchmark and split

`training/dmd/phone_benchmark.py` now rejects incomplete truth. A positive image needs original
dimensions, SHA-256, license, context, human approval and in-bounds pixel boxes. A zero-box image
is accepted only when a human confirms the `ALL_VISIBLE_PHONES` annotation scope. Driver,
passenger and unknown associations remain distinct.

Current benchmark status: **BLOCKED_ZERO_APPROVED_RECORDS**. Positive images: 0; reviewed negatives: 0;
phone boxes: 0. AP calculation is therefore prohibited.

`training/split_dataset.py` now builds connected components across SHA, near-duplicate cluster,
subject, video, session and original asset before assigning a split. It blocks `gZ-37` and
`gE-28` before materialization and runs a second preflight. Current split status is
**NOT_ESTABLISHED_ZERO_APPROVED_RECORDS** because approved records are zero; group-disjoint validation is not claimed.
