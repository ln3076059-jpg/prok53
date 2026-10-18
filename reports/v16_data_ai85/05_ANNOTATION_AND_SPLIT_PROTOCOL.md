# Annotation and split protocol

`training/data_governance.py` defines the requested phone visibility, phone action, context,
occupant-role and three-state seatbelt vocabularies. Every canonical annotation retains source
ID, asset ID, source hash, original label, mapping version, review state and license reference.
Video intervals require a real video ID and valid start/end times; static images cannot acquire
synthetic event timestamps. Positive PHONE_BOX rows require valid normalized coordinates.

Approval requires an authenticated reviewer. UNKNOWN is never converted to SAFE_DRIVING,
visible phone evidence is not automatically phone use, and invisible belt evidence remains
UNCERTAIN_OR_OCCLUDED. Human QA requires double annotation on a sampled subset, disagreement
adjudication and immutable review receipts.

The split validator checks exact content, source group, effective group, subject, video and
original-asset lineage across splits. Augmentations must follow their original asset. `gZ-37`
and `gE-28` are forbidden in train or validation. Because there are zero approved governed
records, no train/validation/locked-test split is claimed as established.

Planning target: group-disjoint data from at least 20 new consenting participants when feasible,
with roughly 100 independently held-out phone events plus enough reviewed negative driving time
to estimate false alarms/minute. These are collection targets, not current sample counts.
