# V2 Data Diversity Audit

Overall status: **NOT_GOVERNED**

This report uses declared manifest metadata and measured image dimensions only. Missing identity or condition metadata is `UNKNOWN`; subject separation is not inferred.

## PHONE_DETECTOR_PROPOSAL_DATA

- Samples: **9728**
- Governance: **NOT_GOVERNED**
- Human review states: `{'PENDING': 9728}`
- Splits: `{'test': 1109, 'train': 7513, 'val': 1106}`
- Source/provider: `{'status': 'DECLARED', 'present_samples': 9728, 'missing_samples': 0, 'unique_values': 1, 'distribution': {'kaggle_habbas_dms_v1': 9728}}`
- Declared source groups: **3616**
- Resolution status: **MEASURED**; measured 9728, unknown 0, unique 460
- Top measured resolutions: `{'640x360': 3618, '640x384': 1607, '640x640': 1024, '640x480': 874, '640x523': 618, '640x426': 299, '480x640': 279, '360x640': 102, '640x427': 87, '640x425': 66}`
- Subject-disjoint: **NOT_PROVABLE**

### Identity and semantic metadata

| Dimension | Status | Present | Missing | Unique |
|---|---:|---:|---:|---:|
| source_id | DECLARED | 9728 | 0 | 1 |
| camera_id | UNKNOWN | 0 | 9728 | 0 |
| video_id | UNKNOWN | 0 | 9728 | 0 |
| clip_id | UNKNOWN | 0 | 9728 | 0 |
| vehicle_id | UNKNOWN | 0 | 9728 | 0 |
| person_id | UNKNOWN | 0 | 9728 | 0 |
| vehicle_type | UNKNOWN | 0 | 9728 | 0 |
| occupant_role | UNKNOWN | 0 | 9728 | 0 |

### Requested condition coverage

| Condition | Status | Declared samples |
|---|---:|---:|
| day | UNKNOWN | 0 |
| night | UNKNOWN | 0 |
| low_light | UNKNOWN | 0 |
| rain | UNKNOWN | 0 |
| glare | UNKNOWN | 0 |
| reflection | UNKNOWN | 0 |
| blur | UNKNOWN | 0 |
| occlusion | UNKNOWN | 0 |
| dark_windshield | UNKNOWN | 0 |
| camera_angle | UNKNOWN | 0 |

### Leakage evidence

| Dimension | Status | Present | Unique | Cross-split overlaps |
|---|---:|---:|---:|---:|
| sha256 | PASS | 9728 | 9728 | 0 |
| phash | PASS | 9728 | 7987 | 0 |
| source_group_id | PASS | 9728 | 3616 | 0 |
| effective_group_id | PASS | 9728 | 3218 | 0 |
| near_cluster_id | PASS | 9728 | 3218 | 0 |
| video_id | NOT_PROVABLE | 0 | 0 | 0 |
| clip_id | NOT_PROVABLE | 0 | 0 | 0 |
| vehicle_id | NOT_PROVABLE | 0 | 0 | 0 |
| person_id | NOT_PROVABLE | 0 | 0 | 0 |
| camera_id | NOT_PROVABLE | 0 | 0 | 0 |

## UPPER_BODY_ROI_PROPOSAL_DATA

- Samples: **4868**
- Governance: **NOT_GOVERNED**
- Human review states: `{'PENDING': 4868}`
- Splits: `{'test': 621, 'train': 3604, 'val': 643}`
- Source/provider: `{'status': 'DECLARED', 'present_samples': 4868, 'missing_samples': 0, 'unique_values': 1, 'distribution': {'roboflow_seatbelttraining_v4': 4868}}`
- Declared source groups: **2750**
- Resolution status: **MEASURED**; measured 4868, unknown 0, unique 1
- Top measured resolutions: `{'416x416': 4868}`
- Subject-disjoint: **NOT_PROVABLE**

### Identity and semantic metadata

| Dimension | Status | Present | Missing | Unique |
|---|---:|---:|---:|---:|
| source_id | DECLARED | 4868 | 0 | 1 |
| camera_id | UNKNOWN | 0 | 4868 | 0 |
| video_id | UNKNOWN | 0 | 4868 | 0 |
| clip_id | UNKNOWN | 0 | 4868 | 0 |
| vehicle_id | UNKNOWN | 0 | 4868 | 0 |
| person_id | UNKNOWN | 0 | 4868 | 0 |
| vehicle_type | UNKNOWN | 0 | 4868 | 0 |
| occupant_role | UNKNOWN | 0 | 4868 | 0 |

### Requested condition coverage

| Condition | Status | Declared samples |
|---|---:|---:|
| day | UNKNOWN | 0 |
| night | UNKNOWN | 0 |
| low_light | UNKNOWN | 0 |
| rain | UNKNOWN | 0 |
| glare | UNKNOWN | 0 |
| reflection | UNKNOWN | 0 |
| blur | UNKNOWN | 0 |
| occlusion | UNKNOWN | 0 |
| dark_windshield | UNKNOWN | 0 |
| camera_angle | UNKNOWN | 0 |

### Leakage evidence

| Dimension | Status | Present | Unique | Cross-split overlaps |
|---|---:|---:|---:|---:|
| sha256 | PASS | 4868 | 4868 | 0 |
| phash | PASS | 4868 | 4260 | 0 |
| source_group_id | PASS | 4868 | 2750 | 0 |
| effective_group_id | PASS | 4868 | 1838 | 0 |
| near_cluster_id | PASS | 4868 | 1838 | 0 |
| video_id | NOT_PROVABLE | 0 | 0 | 0 |
| clip_id | NOT_PROVABLE | 0 | 0 | 0 |
| vehicle_id | NOT_PROVABLE | 0 | 0 | 0 |
| person_id | NOT_PROVABLE | 0 | 0 | 0 |
| camera_id | NOT_PROVABLE | 0 | 0 | 0 |

## SEATBELT_CLASSIFIER_PROPOSAL_DATA

- Samples: **4929**
- Governance: **NOT_GOVERNED**
- Human review states: `{'UNKNOWN': 4929}`
- Splits: `{'test': 624, 'train': 3660, 'val': 645}`
- Source/provider: `{'status': 'UNKNOWN', 'present_samples': 0, 'missing_samples': 4929, 'unique_values': 0, 'distribution': {}}`
- Declared source groups: **2750**
- Resolution status: **MEASURED**; measured 4929, unknown 0, unique 3288
- Top measured resolutions: `{'416x207': 18, '416x171': 12, '416x191': 12, '416x180': 11, '216x416': 10, '416x200': 10, '416x209': 10, '209x416': 9, '416x177': 9, '199x416': 8}`
- Subject-disjoint: **NOT_PROVABLE**

### Identity and semantic metadata

| Dimension | Status | Present | Missing | Unique |
|---|---:|---:|---:|---:|
| source_id | UNKNOWN | 0 | 4929 | 0 |
| camera_id | UNKNOWN | 0 | 4929 | 0 |
| video_id | UNKNOWN | 0 | 4929 | 0 |
| clip_id | UNKNOWN | 0 | 4929 | 0 |
| vehicle_id | UNKNOWN | 0 | 4929 | 0 |
| person_id | UNKNOWN | 0 | 4929 | 0 |
| vehicle_type | UNKNOWN | 0 | 4929 | 0 |
| occupant_role | UNKNOWN | 0 | 4929 | 0 |

### Requested condition coverage

| Condition | Status | Declared samples |
|---|---:|---:|
| day | UNKNOWN | 0 |
| night | UNKNOWN | 0 |
| low_light | UNKNOWN | 0 |
| rain | UNKNOWN | 0 |
| glare | UNKNOWN | 0 |
| reflection | UNKNOWN | 0 |
| blur | UNKNOWN | 0 |
| occlusion | UNKNOWN | 0 |
| dark_windshield | UNKNOWN | 0 |
| camera_angle | UNKNOWN | 0 |

### Leakage evidence

| Dimension | Status | Present | Unique | Cross-split overlaps |
|---|---:|---:|---:|---:|
| sha256 | NOT_PROVABLE | 0 | 0 | 0 |
| phash | NOT_PROVABLE | 0 | 0 | 0 |
| source_group_id | PASS | 4929 | 2750 | 0 |
| effective_group_id | PASS | 4929 | 1838 | 0 |
| near_cluster_id | NOT_PROVABLE | 0 | 0 | 0 |
| video_id | NOT_PROVABLE | 0 | 0 | 0 |
| clip_id | NOT_PROVABLE | 0 | 0 | 0 |
| vehicle_id | NOT_PROVABLE | 0 | 0 | 0 |
| person_id | NOT_PROVABLE | 0 | 0 | 0 |
| camera_id | NOT_PROVABLE | 0 | 0 | 0 |

## Scientific claim boundary

- Subject-disjoint: **NOT_PROVABLE**
- External-domain generalization: **NOT_RUN**
- Human-approved ground truth: **False**
