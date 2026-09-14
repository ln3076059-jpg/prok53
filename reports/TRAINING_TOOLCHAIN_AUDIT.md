# Roadwatch Training Toolchain Audit & Pipeline Lineage

**Document ID:** `ROADWATCH_TRAINING_TOOLCHAIN_AUDIT_V2`  
**Audit Timestamp:** September 13, 2026  
**Governance Policy:** Transparent classification of all `training/` scripts without blind re-execution.  
**Repository:** `https://github.com/ln3076059-jpg/prok53`  

---

## 1. End-to-End Verified Pipeline Execution Chain

The Roadwatch project achieved model training, component validation, and temporal evaluation through the following verified execution chain:

```
[1. Dataset Preparation & Audit]
    training/ingest_dataset.py
    training/split_dataset.py (mc_bootstrap_v2_6500)
    training/audit_mc_bootstrap.py
    training/audit_v2_diversity.py
            ↓
[2. Data Curation & Visual Review]
    training/build_pretrain_pending_v2.py
    training/review_pending_approval.py
    training/finalize_uncertain_visual_review.py
            ↓
[3. Model Component Training]
    training/train_multimodel_v2.py (RTX 3090, 50/50/40 epochs)
    training/epoch_snapshots.py
            ↓
[4. Component Validation]
    training/evaluate_v2.py (Independent Validation Split)
    reports/V2_TRAINING_RESULTS.json
            ↓
[5. Model Freezing & Lock]
    training/lock_model.py (SHA256 manifests)
    models/locked/v2_baseline_001/model_lock.json
            ↓
[6. Canonical Frozen Component Test (Run Count = 1)]
    training/freeze_test.py (Executed exactly once; permanently locked)
    FINAL_PROJECT_REPORT.md Section 11
            ↓
[7. External DMD Stream Acquisition]
    training/dmd/download_curl_extract.py
    training/dmd/adapter.py (OpenLABEL ASAM VCD parser)
    datasets/external_dmd/manifests/dmd_sources.jsonl
            ↓
[8. DMD Temporal Calibration]
    training/dmd/calibrate.py (Development Pool: gC-14, gZ-36)
    training/dmd/temporal_eval.py (Hysteresis & Occlusion Bridging)
    models/temporal_config_v2.json (deep_bridge_40)
            ↓
[9. Final Held-Subject Benchmark 002 (One-Shot)]
    training/dmd/run_benchmark_v2.py (Untouched Holdout: gZ-37)
    reports/DMD_PHONE_TEMPORAL_BENCHMARK_V2.json / .md
            ↓
[10. Publication Figure Generation]
    training/dmd/generate_thesis_figures.py
    reports/figures/ (fig1 through fig4)
            ↓
[11. V3 Multi-View Development & Ablation]
    training/dmd/multiview.py (Synchronous BODY+FACE+HANDS grab)
    training/dmd/pose_features.py (16D geometric pose features)
    training/dmd/fusion.py (Multi-view fusion head)
    training/dmd/run_v3_ablation.py (Development Pool: gC-14, gZ-36, gB-9)
            ↓
[12. Pre-Holdout Freeze & Benchmark 003 Execution]
    reports/DMD_V3_PRE_HOLDOUT_FREEZE.json (Formal code/weight freeze)
    training/dmd/holdout_guard.py (HoldoutAccessError gatekeeper)
    training/dmd/v3_evaluator.py (One-shot run on held gE-28)
    reports/DMD_PHONE_TEMPORAL_BENCHMARK_V3.json / .md
            ↓
[13. V3.1 Precision Recovery Development]
    training/dmd/extract_dev_features.py (Development pool feature cache)
    training/dmd/fusion.py (Auxiliary rescue gating & consensus)
    training/dmd/action_model.py (Pose negative filtering)
    training/dmd/v3_evaluator.py (Cross-view deduplication & fragment merge)
```

---

## 2. Toolchain Classification Matrix

| Script Path | Governance Category | Functional Role in Pipeline |
|---|---|---|
| `training/train_multimodel_v2.py` | `TRAINING` | Unified multi-model training runner on RTX 3090 GPU |
| `training/epoch_snapshots.py` | `TRAINING` | Periodic checkpoint hashing and weights export |
| `training/lock_model.py` | `TRAINING` | Freezes weights into `models/locked/v2_baseline_001/` |
| `training/evaluate_v2.py` | `DEVELOPMENT EVALUATION` | Component validation metric calculation |
| `training/freeze_test.py` | `FINAL HOLDOUT EVALUATION` | Canonical frozen test harness (executed once, preserved) |
| `training/dmd/calibrate.py` | `CALIBRATION` | Multi-parameter hysteresis sweep on development pool |
| `training/dmd/temporal_eval.py` | `CALIBRATION` | Core temporal matching engine and metric calculator |
| `training/dmd/run_benchmark_v2.py` | `FINAL HOLDOUT EVALUATION` | One-shot benchmark runner on held Subject 37 (Benchmark 002) |
| `training/dmd/multiview.py` | `DEVELOPMENT EVALUATION` | Multi-stream video packet grab and frame alignment |
| `training/dmd/pose_features.py` | `DEVELOPMENT EVALUATION` | 16-D scale-normalized anatomical feature extractor |
| `training/dmd/fusion.py` | `DEVELOPMENT EVALUATION` | Multi-view evidence fusion head with rescue gating |
| `training/dmd/run_v3_ablation.py` | `DEVELOPMENT EVALUATION` | Subject-disjoint ablation on development pool (gC-14, gZ-36, gB-9) |
| `training/dmd/extract_dev_features.py`| `DEVELOPMENT EVALUATION` | Feature caching for rapid development ablation |
| `training/dmd/holdout_guard.py` | `GOVERNANCE_GATE` | Programmatic block preventing access to holdouts (gZ-37, gE-28) |
| `training/dmd/v3_evaluator.py` | `DEVELOPMENT EVALUATION` / `FINAL HOLDOUT EVALUATION` | Execution engine for A0–A7 / B0–B6 and Benchmark 003 |
| `training/dmd/run_v3_master_pipeline.py` | `FINAL HOLDOUT EVALUATION` | One-shot Benchmark 003 orchestrator on gE-28 |
| `training/dmd/adapter.py` | `DEVELOPMENT EVALUATION` | OpenLABEL ASAM VCD annotation parser for DMD |
| `training/dmd/generate_thesis_figures.py` | `REPORTING` | Publication figure generator for thesis artifact |
| `training/dmd/evaluate_temporal.py` | `HISTORICAL` | Predecessor benchmark runner (Benchmark 001 on Sub 36) |
| `training/dmd/download_curl_extract.py` | `DATA_ACQUISITION` | Rate-limited archive downloader and stream extractor |
| `training/ingest_dataset.py` | `DATA_AUDIT` | Parses raw images and ground truth bounding boxes |
| `training/split_dataset.py` | `DATA_AUDIT` | Creates subject/source-disjoint train/val/test splits |
| `training/audit_mc_bootstrap.py` | `DATA_AUDIT` | Verifies image integrity and detects cross-split leakage |
| `training/audit_v2_diversity.py` | `DATA_AUDIT` | Audits lighting, camera angle, and demographic diversity |
| `training/build_pretrain_pending_v2.py`| `REVIEW` | Identifies low-confidence or ambiguous annotations |
| `training/review_pending_approval.py` | `REVIEW` | Human/AI review triage for candidate data samples |
| `training/finalize_uncertain_visual_review.py` | `REVIEW` | Encodes confirmed annotations into training datasets |

---

## 3. Governance Conclusion

Every script in the active toolchain serves a verified role in the training, calibration, and evaluation provenance chain. Consumed holdouts (`gZ-37`, `gE-28`) are programmatically protected by `holdout_guard.py` and are permanently forbidden from development use. Future evaluation on unseen data (`BENCHMARK_004`) remains strictly blocked until a new untouched DMD subject is provisioned.
