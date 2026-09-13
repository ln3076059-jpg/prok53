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
```

---

## 2. Toolchain Classification Matrix

| Script Path | Primary Classification | Functional Role in Final Pipeline |
|---|---|---|
| `training/train_multimodel_v2.py` | `USED_IN_FINAL_MODEL_TRAINING` | Unified multi-model training runner on RTX 3090 GPU |
| `training/epoch_snapshots.py` | `USED_IN_FINAL_MODEL_TRAINING` | Periodic checkpoint hashing and weights export |
| `training/evaluate_v2.py` | `USED_IN_FINAL_MODEL_TRAINING` | Component validation metric calculation |
| `training/lock_model.py` | `USED_IN_FINAL_MODEL_TRAINING` | Freezes weights into `models/locked/v2_baseline_001/` |
| `training/ingest_dataset.py` | `USED_IN_DATA_AUDIT` | Parses raw images and ground truth bounding boxes |
| `training/split_dataset.py` | `USED_IN_DATA_AUDIT` | Creates subject/source-disjoint train/val/test splits |
| `training/audit_mc_bootstrap.py` | `USED_IN_DATA_AUDIT` | Verifies image integrity and detects cross-split leakage |
| `training/audit_v2_diversity.py` | `USED_IN_DATA_AUDIT` | Audits lighting, camera angle, and demographic diversity |
| `training/audit_v2_readiness.py` | `USED_IN_DATA_AUDIT` | Pre-training checklist and gatekeeper |
| `training/build_pretrain_pending_v2.py`| `USED_IN_REVIEW` | Identifies low-confidence or ambiguous annotations |
| `training/review_pending_approval.py` | `USED_IN_REVIEW` | Human/AI review triage for candidate data samples |
| `training/finalize_uncertain_visual_review.py` | `USED_IN_REVIEW` | Encodes confirmed annotations into training datasets |
| `training/freeze_test.py` | `USED_IN_FINAL_MODEL_TRAINING` | Canonical frozen test harness (executed once, preserved) |
| `training/freeze_external_test.py` | `USED_IN_FINAL_MODEL_TRAINING` | Verification of external test set boundaries |
| `training/dmd/adapter.py` | `USED_IN_DMD_BENCHMARK` | OpenLABEL ASAM VCD annotation parser for DMD |
| `training/dmd/calibrate.py` | `USED_IN_TEMPORAL_CALIBRATION` | Multi-parameter hysteresis sweep on development pool |
| `training/dmd/temporal_eval.py` | `USED_IN_TEMPORAL_CALIBRATION` | Core temporal matching engine and metric calculator |
| `training/dmd/download_curl_extract.py` | `USED_IN_DMD_BENCHMARK` | Rate-limited archive downloader and stream extractor |
| `training/dmd/extract_s2_when_ready.py` | `USED_IN_DMD_BENCHMARK` | On-the-fly decompressor for target DMD session members |
| `training/dmd/run_benchmark_v2.py` | `USED_IN_DMD_BENCHMARK` | One-shot benchmark runner on held Subject 37 |
| `training/dmd/generate_thesis_figures.py` | `USED_IN_DMD_BENCHMARK` | Publication figure generator for thesis artifact |
| `training/dmd/evaluate_temporal.py` | `HISTORICAL` | Predecessor benchmark runner (Benchmark 001 on Sub 36) |
| `training/dmd/acquire_subject.py` | `HISTORICAL` | Legacy single-threaded subject downloader |
| `training/dmd/download_dmd.py` | `HISTORICAL` | Pre-curl range-request downloader |
| `training/dmd/split.py` | `HISTORICAL` | Initial DMD partition draft |
| `training/dmd/timeline.py` | `HISTORICAL` | Initial video presentation timeline audit utility |
| `training/build_mc_bootstrap.py` | `HISTORICAL` | Legacy bootstrap builder for V1 models |
| `training/build_phone_bootstrap.py` | `HISTORICAL` | Early phone dataset harvester |
| `training/kaggle_runner.py` | `HISTORICAL` | Cloud Kaggle remote training script |
| `training/train_fusion.py` | `HISTORICAL` | V1 logistic regression fusion experiment |
| `training/build_seatbelt_v2.py` | `HISTORICAL` | Early prototype for seatbelt upper-body cropping |
| `training/export_onnx.py` | `NOT_REQUIRED` | ONNX edge conversion utility (optional deployment) |
| `training/download_base_weights.py` | `NOT_REQUIRED` | Baseline Ultralytics weight fetcher |

---

## 3. Governance Conclusion

Every script in the active toolchain serves a verified role in the training, calibration, and evaluation provenance chain. Historical scripts are preserved for academic reproducibility but are strictly barred from re-execution against frozen holdouts.
