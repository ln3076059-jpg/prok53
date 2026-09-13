# Roadwatch Final Project Truth Audit

**Audit Timestamp:** 2026-09-13T02:00:00Z  
**Base Repository HEAD:** `fc6cfa9fde7f4f3292749bd1564e9af7f81b6281`  
**Auditor:** Automated Engineering Completion Pipeline  
**Governance Mode:** DMD-First Temporal Development  

---

## 1. Evidence Priority Hierarchy
All statements in project documentation are evaluated strictly according to the governance hierarchy:
1. **Actual Model Files & Weights** (`models/locked/v2_baseline_001/`)
2. **SHA256 Manifests** (`models/locked/v2_baseline_001/model_lock.json`)
3. **Model Lock & Environment Records** (`reports/V2_TRAINING_RESULTS.json`)
4. **Raw Experiment Outputs & Execution Logs**
5. **Generated Intermediate Reports**
6. **Prose in Documentation** (`README.md`, `PRODUCT.md`)

Where prose contradicts immutable artifacts on disk, the immutable artifact is the sole scientific truth, and the prose is marked `STALE_DOCUMENTATION` and reconciled.

---

## 2. Component Model Artifacts Audit

| Component | Weight Path | File Size (Bytes) | SHA256 Hash | Status |
|---|---|---|---|---|
| **Phone Detector** | `models/locked/v2_baseline_001/phone_detector/best.pt` | 19,209,626 | `840a29cb2151b881279cdabe25b03b28c5dcf40a43464edd6b672c8851f77d54` | **VERIFIED_CURRENT** |
| **Seatbelt Detector** | `models/locked/v2_baseline_001/seatbelt_detector/best.pt` | 19,345,306 | `361436ab073c8fcc17a041f098285efdd0cf6775a8970be2ffada4e45c6bc500` | **VERIFIED_CURRENT** |
| **Seatbelt Classifier** | `models/locked/v2_baseline_001/seatbelt_classifier/best.pt` | 11,028,162 | `e55158e3f152922e710ad260295da16488ffc2c6a145b8dfdc5c4ce323392bd4` | **VERIFIED_CURRENT** |
| **Vehicle Detector (Aux)** | `models/auxiliary/yolo11n.pt` | 5,613,764 | `3c8d3568c4d293ca5610bc13328e08d6c7075c32ab5fa626b9c9baeeffc24151` | **VERIFIED_CURRENT** |
| **Pose Estimator (Aux)** | `models/auxiliary/yolo11n-pose.pt` | 6,255,593 | `fa3eb05c4889c2598380d19a2e3cfcf1a9528bb5eb4fbaf9b9e67d2ca85806c9` | **VERIFIED_CURRENT** |

**Audit Result:** All component model weights exist on disk, are non-empty, and their SHA256 checksums match `models/locked/v2_baseline_001/model_lock.json` with 100% byte-exact precision.

---

## 3. Training & Component Validation Metrics Audit

Evidence source: `reports/V2_TRAINING_RESULTS.json` (Experiment: `MULTIMODEL_V2_RTX5060TI_16GB_ALL_PRETRAIN_PENDING_TRAIN`, trained on NVIDIA RTX 3090, PyTorch 2.8.0+cu128).

| Model Component | Metric | Recorded Metric Value | Evidence Status |
|---|---|---|---|
| **Phone Detector** | Precision (B) | 97.13% | **VERIFIED_CURRENT** |
| | Recall (B) | 82.73% | **VERIFIED_CURRENT** |
| | mAP50 (B) | 94.48% | **VERIFIED_CURRENT** |
| | mAP50-95 (B) | 70.03% | **VERIFIED_CURRENT** |
| **Seatbelt Detector** | Precision (B) | 92.57% | **VERIFIED_CURRENT** |
| | Recall (B) | 87.85% | **VERIFIED_CURRENT** |
| | mAP50 (B) | 94.73% | **VERIFIED_CURRENT** |
| | mAP50-95 (B) | 53.70% | **VERIFIED_CURRENT** |
| **Seatbelt Classifier** | Accuracy Top-1 | 80.97% (~81.0%) | **VERIFIED_CURRENT** |
| | Accuracy Top-5 | 100.0% | **VERIFIED_CURRENT** |

---

## 4. Frozen Component Test Audit

Evidence sources: `reports/frozen_test/V2_FROZEN_TEST_RESULTS.json`, `FINAL_PROJECT_REPORT.md` Section E.

| Model Component | Metric | Value | Verification Finding |
|---|---|---|---|
| **Phone Detector (Test)** | Precision / Recall / mAP50 / mAP50-95 | 90.2% / 86.2% / 90.5% / 65.5% | **VERIFIED_HISTORICAL** |
| **Seatbelt Detector (Test)** | Precision / Recall / mAP50 / mAP50-95 | 93.8% / 88.2% / 92.7% / 47.7% | **VERIFIED_HISTORICAL** |
| **Seatbelt Classifier (Test)** | Top-1 Accuracy | 78.9% | **VERIFIED_HISTORICAL** |
| **Frozen Test Run Count** | Execution count | **1** | **PRESERVED_NOT_RERUN** |

Per project governance: The canonical frozen model test was executed once and is permanently locked. It must **never** be rerun.

---

## 5. Documentation Contradiction Reconciliation

| Document | Statement | Truth Finding | Corrective Action |
|---|---|---|---|
| `README.md` L58-59 | *"Status: NOT_RUN. No trained model exists."* | **STALE_DOCUMENTATION** (Contradicts verified weights in `models/locked/`) | Update to report verified V2 baseline weights, metrics, and locked status |
| `README.md` L83 | *"Current V2 status is UNTRAINED / NOT APPROVED"* | **STALE_DOCUMENTATION** | Update to reflect: Trained baseline verified; DMD temporal development active; production certification false |
| `PRODUCT.md` L44 | *"trained weights, and measured metrics are not yet available"* | **STALE_DOCUMENTATION** | Update to reflect verified V2 multi-model pipeline |
| `FINAL_PROJECT_REPORT.md` | Baseline evidence verified; event gates open | **VERIFIED_CURRENT** | Synchronize README and PRODUCT to tell the identical scientific story |

---

## 6. Audit Conclusion
The repository possesses real, fully trained, cryptographically verified V2 baseline weights and component validation records. The historical documentation claiming "no trained model exists" was out-of-date and has been reconciled. Temporal phone evaluation proceeds via DMD continuous distraction videos.
