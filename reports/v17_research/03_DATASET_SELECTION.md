# 03. Dataset Selection & Data Governance

## 1. Current Inventory Audit (28,555 Rows)
An exhaustive scan of `dataset/manifests/` reveals:
1. `ingest_adt_seatbelt_v1.jsonl`: 2,704 rows (0 approved, pending provenance review).
2. `ingest_anywaylabs_synthetic_dms.jsonl`: 1,356 rows (0 approved, synthetic domain shift unverified).
3. `ingest_c3rl_seatbelt_v1.jsonl`: 5,000 rows (0 approved, synthetic geometric images).
4. `ingest_dms.jsonl`: 9,884 rows (0 approved, 156 missing assets, license pending).
5. `ingest_mendeley_driver_risk_v1.jsonl`: 1,232 rows (0 approved, 5 driver groups only).
6. `ingest_roboflow_seatbelttraining_v4.jsonl`: 8,371 rows (0 approved, 8,371 pending reviews).
7. `ingest_sintes.jsonl`: 8 rows (0 approved, single subject).

**Total Approved Training Rows:** **0**.

## 2. Priority Dataset Action Plan
| Dataset | Official Source | Exact Action & Link | Allowed Post-Approval | Expected Contribution |
|---|---|---|---|---|
| **DMD (Vicomtech)** | [GitHub](https://github.com/Vicomtech/DMD-Driver-Monitoring-Dataset) | Submit Gated Portal Form at [opendatasets.vicomtech.org](https://opendatasets.vicomtech.org/) | Video temporal diagnostics & fitting | Ground truth temporal phone call / texting events |
| **100-Driver** | [PIC Lab](https://100-driver.github.io/) | Sign & email [100_Driver_Licensing.pdf](https://100-driver.github.io/img/100_Driver_Licensing.pdf) | Cross-driver action representation fitting | High-diversity driver action generalization (100 subjects) |
| **State Farm** | [Kaggle](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/data) | Accept rules at [Kaggle Rules](https://www.kaggle.com/competitions/state-farm-distracted-driver-detection/rules) | Static image action classification | SAFE vs PHONE vs OTHER classification pretraining |
| **Mendeley Risk** | [Mendeley](https://data.mendeley.com/datasets/562zj8n7xf/1) | Direct Download (CC BY 4.0, 138MB) | In-cabin bbox & behavior validation | High-res real car positive/negative frames |
| **Roboflow Candidates** | [Roboflow Universe](https://universe.roboflow.com/) | Provenance & class audit | 3-state seatbelt training | Real vehicle seatbelt fastened/unfastened/uncertain |
