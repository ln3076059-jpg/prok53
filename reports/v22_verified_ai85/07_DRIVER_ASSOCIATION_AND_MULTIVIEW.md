# Driver association and multiview

Fixed pose-present and pose-missing paths that could create driver identity without role evidence. Highest-confidence pose in a crop is now unverified UNKNOWN in both V21 entrypoints. Passenger and verified-mounted rejection tests remain applicable; unverified mounted suggestions remain UNKNOWN.

V21 enable_dynamic_roi was unused; actual crop is static. No calibrated tracking or real association ground-truth set was found. Driver association precision, passenger exclusion, true-driver retention and 99.6% containment cannot be claimed. ROI_ASSOCIATION_METRICS_NOT_REPRODUCIBLE.

multiview_feasibility.json records actual file hashes, FPS and frame counts. gZ-36 has BODY/FACE/HANDS locally; gC-14 has BODY only. Content synchronization, offsets and orientation are not proven by file presence. Full-cohort fusion is blocked pending verified synchronization/calibration and missing views.
