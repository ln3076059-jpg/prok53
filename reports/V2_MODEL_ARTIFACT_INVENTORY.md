# Roadwatch V2 Model Artifact Inventory & Integrity Audit

**Inventory Timestamp:** September 13, 2026  
**Baseline Model Lock:** `V2_BASELINE_001`  
**Lock Manifest:** `models/locked/v2_baseline_001/model_lock.json`  
**Validation Report:** `reports/V2_TRAINING_RESULTS.json`  
**Canonical Frozen Test Report:** `FINAL_PROJECT_REPORT.md` (Section 11, Run Count = 1)  
**Governance Standard:** Immutable cryptographic verification. No retraining without empirical necessity.  

```ini
MODEL_ARTIFACTS_VERIFIED = true
FROZEN_TEST_RUN_COUNT = 1
CANONICAL_FROZEN_MODEL_TEST = PRESERVED_NOT_RERUN
```

---

## 1. Active Multi-Model Pipeline Artifacts

| Component Name | Architecture | Active Weight Path | SHA256 Checksum | Model Lock Entry | Training Experiment ID | Validation Report Reference | Frozen Test Report Reference | Operational Status |
|---|---|---|---|---|---|---|---|---|
| **Phone Detector** | Ultralytics YOLO11s (detect) | `models/active/v2/phone_detector.pt` | `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54` | `models/locked/v2_baseline_001/phone_detector/best.pt` | `MULTIMODEL_V2_RTX5060TI_16GB_ALL_PRETRAIN_PENDING_TRAIN` | `reports/V2_TRAINING_RESULTS.json` (P 97.13%, R 82.73%, mAP50 94.48%) | `FINAL_PROJECT_REPORT.md` Sec. 11 (P 90.20%, R 86.20%, mAP50 90.50%) | **VERIFIED_ACTIVE** |
| **Seatbelt Detector** | Ultralytics YOLO11s (detect) | `models/active/v2/seatbelt_detector.pt` | `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500` | `models/locked/v2_baseline_001/seatbelt_detector/best.pt` | `MULTIMODEL_V2_RTX5060TI_16GB_ALL_PRETRAIN_PENDING_TRAIN` | `reports/V2_TRAINING_RESULTS.json` (P 92.57%, R 87.85%, mAP50 94.73%) | `FINAL_PROJECT_REPORT.md` Sec. 11 (P 93.80%, R 88.20%, mAP50 92.70%) | **VERIFIED_ACTIVE** |
| **Seatbelt Classifier** | Ultralytics YOLO11s-cls (classify) | `models/active/v2/seatbelt_classifier.pt` | `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4` | `models/locked/v2_baseline_001/seatbelt_classifier/best.pt` | `MULTIMODEL_V2_RTX5060TI_16GB_ALL_PRETRAIN_PENDING_TRAIN` | `reports/V2_TRAINING_RESULTS.json` (Top-1 80.97%, Top-5 100.0%) | `FINAL_PROJECT_REPORT.md` Sec. 11 (Top-1 78.90%, Top-5 100.0%) | **VERIFIED_ACTIVE** |
| **Vehicle Detector** | YOLO11n (auxiliary) | `models/auxiliary/yolo11n.pt` | `3c8d3568c4d293ca5610bc13328e08d6c7075c32ab5fa626b9c9baeeffc24151` | `models/auxiliary/yolo11n.pt` | Ultralytics COCO Baseline | COCO Val Baseline | Not Part of Safety Task Test | **VERIFIED_AUXILIARY** |
| **Pose Estimator** | YOLO11n-pose (auxiliary) | `models/auxiliary/yolo11n-pose.pt` | `fa3eb05c4889c2598380d19a2e3cfcf1a9528bb5eb4fbaf9b9e67d2ca85806c9` | `models/auxiliary/yolo11n-pose.pt` | Ultralytics COCO Pose Baseline | COCO Keypoints Val | Not Part of Safety Task Test | **VERIFIED_AUXILIARY** |

---

## 2. Configuration & Temporal Artifacts

| Configuration Artifact | Path | SHA256 Checksum | Role / Description | Status |
|---|---|---|---|---|
| **Model Config V2** | `models/model_config_v2.yaml` | `7755de7b47ba57cd55e7bca86895e998710386103844cb4be7dd67106f02be63` | Detection thresholds, driver ROI box, seatbelt 3-class policy | **FROZEN_ACTIVE** |
| **Temporal Config V2** | `models/temporal_config_v2.json` | `191918e41e5715008a3a90c03aa062d86c4fe7086401f55c2fa55d5d1c189b4f` | `deep_bridge_40` parameter set (bridge 4.0s, act 0.25, rel 0.15) | **FROZEN_ACTIVE** |
| **Matching Policy** | `reports/DMD_EVENT_MATCHING_POLICY_FREEZE.json` | `6b9fa96924f79ba65ff29672eb29b3503aa6981881515ef98ecb2f7dd06ffdd9` | Standard temporal action detection matching rules | **FROZEN_ACTIVE** |

---

## 3. Cryptographic Verification Statement

All active weights have been directly re-hashed using SHA-256 on the local filesystem. Every checksum precisely matches the immutable baseline lock `models/locked/v2_baseline_001/model_lock.json`. No retraining or weights drift has occurred.
