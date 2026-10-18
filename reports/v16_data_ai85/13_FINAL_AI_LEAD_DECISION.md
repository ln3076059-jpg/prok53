# Final AI lead decision

**Decision: REJECT production promotion and reject an AI85 claim.**

V16 verified all nine requested dataset candidates and strengthened the acquisition and
annotation boundary, but zero sources currently satisfy all rights, provenance, human-label and
group-split requirements. The existing inventory has 28555 rows and zero
approved training rows. No V16 media was downloaded, no protected subject was accessed, no
model was fitted, and no active checkpoint or event logic changed.

The tiled detector is technically implemented but has not demonstrated an AP improvement. Phone
event recall/F1, SAFE recall, seatbelt Macro-F1 and FA/min are unchanged. Independent AI85 is
not established.

Next admissible work is: obtain written access for a selected source; record license and allowed
operations in the registry; review provenance; annotate real positive boxes and scoped negatives;
double-review a QA subset; build a subject/session/original-asset-disjoint split; lock the test;
then run D0-D4 before any fitting. Seatbelt work additionally needs completion of the 500 pending
human reviews. Production approval remains a separate human decision.

Readiness snapshot: `{"action_fitting": "BLOCKED_RIGHTS_AND_GROUP_SPLIT", "approved_training_data": "BLOCKED_ZERO_APPROVED_RECORDS", "association_evaluation": "BLOCKED_NO_VERIFIED_ROLE_GT", "dataset_registry": "READY", "download_gate": "READY_FAIL_CLOSED", "event_replay": "BLOCKED_NO_NEW_VALIDATED_CANDIDATE", "independent_evaluation": "BLOCKED_NO_LOCKED_TEST", "phone_detection_ablation": "BLOCKED_NO_APPROVED_BBOX_POSITIVE_NEGATIVE_VALIDATION", "production": "REJECT", "seatbelt_fitting": "BLOCKED_500_PENDING_LABELS", "unified_annotation_schema": "READY_ENGINEERING"}`
