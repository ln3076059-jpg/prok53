# 06. Full Source Code Audit
Audited 12 core modules:
- `backend/ai/detector.py`: Single process-wide YOLO instance, strict class contract (`phone`, `seatbelt_fastened`, `seatbelt_unfastened`).
- `backend/ai/events.py`: `TemporalEventEngine` hysteresis state machine with multi-observation gating and cooldown.
- `training/dmd/online_candidate.py`: Online E3 streaming features and `ResearchCandidate` adapter.
- `training/dmd/temporal_eval.py`: Continuous sequence evaluation with IoU and onset tolerance matching.
- `training/dmd/sliced_phone.py`: SAHI-inspired source-resolution tiled inference.
- `training/split_dataset.py`: Graph-connected subject group split with leakage isolation.
- `training/data_governance.py`: Annotation governance and protected subject enforcement.
