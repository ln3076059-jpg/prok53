# 02. Repository & Infrastructure Audit

## 1. System Metadata
* **Repository:** `ln3076059-jpg/prok53`
* **Active Branch:** `roadwatch-v4-candidate`
* **Local Head:** `78dd0078ffd05bc23b41806162341b7e77c9cb48`
* **CI Test Suite:** 654 tests passing (target 639 surpassed with 15 new recovery tests).
* **Code Quality:** Compile clean (`compileall` PASS), Linter clean (`ruff` PASS), Frontend build clean (`vite v7.3.6` PASS), Backend smoke clean (FastAPI 17 endpoints PASS).

## 2. Directory Layout & Architecture
* `backend/ai/`: Production and candidate inference engines:
  * `detector.py`: Normalized YOLO detection wrapper.
  * `association.py`: Occupant spatial association and cabin geometry.
  * `fusion.py`: Multi-feature logistic fusion with fail-closed fallback.
  * `events.py`: Temporal event persistence engine (`TemporalEventEngine`).
* `training/dmd/`: Research modules and validation toolchains:
  * `sliced_phone.py`: Source-resolution SAHI-inspired tiled detector adapter.
  * `autonomous_phone.py`: Autonomous driver ROI proposal pipeline.
  * `candidate_flow.py`: Stage-level candidate rejection and coverage ledger.
  * `phone_benchmark.py`: Fail-closed benchmark schema for positive/negative validation.
  * `research_permissions.py`: Live fail-closed rights and operation guards.
* `training/split_dataset.py`: Graph-connected component grouping ensuring subject-, video-, session-, and original-asset-disjoint isolation.
* `models/`: Frozen development weights and candidate configurations:
  * `model_config_v4_candidate.yaml`: Candidate config with `activation_threshold: 0.25`, `min_positive_seconds: 0.50s`.
  * `models/active/v2/phone_detector.pt`: Specialist phone detector.
  * `models/auxiliary/yolo11n-pose.pt`: Pose estimation backbone.
