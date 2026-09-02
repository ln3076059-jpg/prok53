# Camera Generalization Audit
known_camera_ids: 0
unknown_camera_ids: 3013
camera_isolation_provable: NOT_PROVABLE
- Detailed check: Since raw datasets lack `camera_id` tags, split leakage is theoretically possible. External testing on novel datasets is required to prove true generalization.
