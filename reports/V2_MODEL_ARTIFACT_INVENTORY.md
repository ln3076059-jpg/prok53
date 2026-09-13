# Roadwatch V2 Model Artifact Inventory

**Inventory Timestamp:** 2026-09-13T02:17:00Z  
**Baseline Model Lock:** `V2_BASELINE_001`  
**Lock Manifest:** `models/locked/v2_baseline_001/model_lock.json`  
**Training Results:** `reports/V2_TRAINING_RESULTS.json`  
**Status:** **ALL ARTIFACTS VERIFIED AND ACTIVE**  

---

## 1. Active Multi-Model Pipeline Artifacts

| Component Role | Architecture | Active Weight Path | Canonical Locked Path | File Size (Bytes) | SHA256 Checksum | Source Experiment | Status |
|---|---|---|---|---|---|---|---|
| **Phone Detector** | YOLO11s (detect) | `models/active/v2/phone_detector.pt` | `models/locked/v2_baseline_001/phone_detector/best.pt` | 19,209,626 | `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54` | `MULTIMODEL_V2_RTX5060TI_16GB_ALL_PRETRAIN_PENDING_TRAIN` | **VERIFIED_ACTIVE** |
| **Seatbelt Detector** | YOLO11s (detect) | `models/active/v2/seatbelt_detector.pt` | `models/locked/v2_baseline_001/seatbelt_detector/best.pt` | 19,345,306 | `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500` | `MULTIMODEL_V2_RTX5060TI_16GB_ALL_PRETRAIN_PENDING_TRAIN` | **VERIFIED_ACTIVE** |
| **Seatbelt Classifier** | YOLO11s-cls (classify) | `models/active/v2/seatbelt_classifier.pt` | `models/locked/v2_baseline_001/seatbelt_classifier/best.pt` | 11,028,162 | `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4` | `MULTIMODEL_V2_RTX5060TI_16GB_ALL_PRETRAIN_PENDING_TRAIN` | **VERIFIED_ACTIVE** |
| **Vehicle Detector** | YOLO11n (detect) | `models/auxiliary/yolo11n.pt` | `models/auxiliary/yolo11n.pt` | 5,613,764 | `3c8d3568c4d293ca5610bc13328e08d6c7075c32ab5fa626b9c9baeeffc24151` | Ultralytics COCO Baseline | **VERIFIED_AUXILIARY** |
| **Pose Estimator** | YOLO11n-pose (pose) | `models/auxiliary/yolo11n-pose.pt` | `models/auxiliary/yolo11n-pose.pt` | 6,255,593 | `fa3eb05c4889c2598380d19a2e3cfcf1a9528bb5eb4fbaf9b9e67d2ca85806c9` | Ultralytics COCO Pose Baseline | **VERIFIED_AUXILIARY** |

---

## 2. Configuration & Manifest Binding

- **Runtime Configuration:** `models/model_config_v2.yaml`
- **Active Weights Root:** `models/active/`
- **Hash Binding Verification:**
  - `phone_detector`: Matches `model_lock.json` hash `840a29cb...`
  - `seatbelt_detector`: Matches `model_lock.json` hash `361436ab...`
  - `seatbelt_classifier`: Matches `model_lock.json` hash `e55158e3...`
  - `yolo11n`: Matches COCO verification hash `3c8d3568...`
  - `yolo11n-pose`: Matches COCO pose verification hash `fa3eb05c...`
- **Frozen Test Run Policy:** Locked at run count **1**. Canonical frozen test is permanently preserved and must never be rerun.
